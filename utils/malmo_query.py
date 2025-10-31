import re
from pathlib import Path
from itertools import cycle

import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from transformers.generation.utils import GenerationMixin


def collect_molmo_points(
    *,
    iteration_working_dir,
    cropped=False,
    seeds,
    use_first_prompt=False,
    n_views,
    prompts,
    ckpt: str = "cyan2k/molmo-7B-D-bnb-4bit",
):
    """
    Query Molmo for pixel coordinates and overlay them on each image.

    Parameters
    ----------
    iteration_working_dir : str | Path
        Root folder that contains sub-dirs named after each seed.
    seeds : Iterable[int]
    n_views : Iterable[int]
        View indices → image names are "camera_angle_{view}.png".
    prompts : list[str]
        Natural-language queries (e.g. “Point at the red cube”).
    ckpt : str, optional
        Hugging Face model identifier or local path.

    Returns
    -------
    dict[int, list[tuple[int, list[tuple[float, float, str]]]]]
        {seed: [(view, [(x_px, y_px, prompt), ...]), …], …}
    """

    # ── 1.  Patch GenerationMixin once ───────────────────────────────────
    if not hasattr(GenerationMixin, "_extract_past_from_model_output"):

        def _extract_past_from_model_output(self, outputs):
            for name in (
                "past_key_values",
                "mems",
                "past_buckets_states",
                "cache_params",
            ):
                cache = getattr(outputs, name, None)
                if cache is not None:
                    return name, cache
            return "past_key_values", None

        GenerationMixin._extract_past_from_model_output = _extract_past_from_model_output

    # ── 2.  Lazy-load / cache Molmo ──────────────────────────────────────
    global _MOLMO_CACHE
    if "_MOLMO_CACHE" not in globals():
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        processor = AutoProcessor.from_pretrained(
            ckpt, trust_remote_code=True, device_map="auto"
        )
        model = AutoModelForCausalLM.from_pretrained(
            ckpt,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bnb_cfg,
        )
        _MOLMO_CACHE = (processor, model)
    else:
        processor, model = _MOLMO_CACHE

    # ── 3.  Main loop ────────────────────────────────────────────────────
    colour_cycle_master = ["red", "lime", "deepskyblue", "orange", "magenta"]
    all_points = {}
    iteration_working_dir = Path(iteration_working_dir)

    for seed in seeds:
        seed_dir = iteration_working_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_points = []

        if prompts is None:
            prompt_file = seed_dir / "prompts.txt"
            if not prompt_file.exists():
                raise FileNotFoundError(
                    f"Expected {prompt_file} but it does not exist."
                )

            # Execute the file in an isolated namespace to recover the variables
            local_ns: dict = {}
            with open(prompt_file, "r", encoding="utf-8") as f:
                exec(f.read(), {}, local_ns)

            # Gather the per‑view lists in the order given by n_views
            prompts = []
            for v in n_views:
                key = f"view{v}_prompts"
                if use_first_prompt is True:
                    key = "view0_prompts"
                if key not in local_ns:
                    raise KeyError(
                        f"Variable '{key}' not found in {prompt_file}. "
                        "Make sure the file defines e.g. 'view0_prompts = [...]'."
                    )
                prompts.append(local_ns[key])
            #load prompts.txt in from seed_dir
            #create a list called prompts= []
            #load in each view0_prompts, view1_prompts, etc. (these are lists that look like "view0_prompts = ["Pint at x","Point at y", etc."]"
            # Fill out the prompts list: prompts = [view0_prompts, view1_prompts, etc.]

        for view in n_views:
            if cropped==False:
                img_path = seed_dir / f"camera_angle_{view}.png"
            else:
                img_path = seed_dir / f"camera_angle_{view}_cropped.png"
            if not img_path.is_file():
                print(f"⚠️  Missing image: {img_path}")
                continue

            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            view_points = []

            view_prompt = prompts[view]

            for prompt in view_prompt:
                if prompt == "Point at the top surface of the cube on the cabinet":
                    print("Changed")
                    prompt = "Point at the cube."

                if prompt == "Point at the top face of the cube with the number 1 printed on it":
                    print("Changed")
                    prompt = "Point at the cube with the number 1 printed on it"
                inputs = processor.process(images=[img], text=prompt)
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

                with torch.autocast(model.device.type, dtype=torch.float16):
                    out = model.generate_from_batch(
                        inputs,
                        GenerationConfig(
                            max_new_tokens=100, stop_strings="<|endoftext|>"
                        ),
                        tokenizer=processor.tokenizer,
                    )

                txt = processor.tokenizer.decode(
                    out[0, inputs["input_ids"].size(1) :], skip_special_tokens=True
                )
                m = re.search(r'x="([\d.]+)".*?y="([\d.]+)"', txt)
                if m:
                    x_norm, y_norm = map(float, m.groups())
                    x_px, y_px = x_norm / 100 * w, y_norm / 100 * h
                    view_points.append((x_px, y_px, prompt))
                else:
                    print(
                        f"⚠️  No <point> tag for '{prompt}' "
                        f"(seed={seed}, view={view})"
                    )
            """
            # ── save overlay ─────────────────────────────────────────────
            fig = plt.figure(figsize=(12, 8 * h / w))
            plt.imshow(img)
            for (x, y, lbl), colour in zip(
                view_points, cycle(colour_cycle_master)
            ):
                plt.scatter(
                    x,
                    y,
                    c=colour,
                    s=80,
                    edgecolors="black",
                    linewidths=0.8,
                    label=lbl,
                )
            plt.axis("off")
            plt.legend(loc="upper right", fontsize=9, framealpha=0.8)
            plt.tight_layout()
            fig.savefig(seed_dir / f"points_view_{view}.png", bbox_inches="tight")
            plt.close(fig)
            """

            # make the figure wider
            fig = plt.figure(figsize=(12, 8 * h / w))
            ax  = fig.add_subplot(111)

            ax.imshow(img)
            for (x, y, lbl), colour in zip(view_points, cycle(colour_cycle_master)):
                ax.scatter(x, y,
                        c=colour,
                        s=80,
                        edgecolors="black",
                        linewidths=0.8,
                        label=lbl)

            ax.axis("off")

            # move the legend outside to the right
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),      # x, y coordinate just outside axes
                borderaxespad=0,               # no padding between axes and legend
                fontsize=9,
                framealpha=0.8
            )

            # reserve space on the right for the legend
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            fig.savefig(seed_dir / f"points_view_{view}.png", bbox_inches="tight")
            plt.close(fig)
            seed_points.append((view, view_points))

        # ── per-seed text summary ───────────────────────────────────────
        with open(seed_dir / "points.txt", "w") as fh:
            for view, pts in seed_points:
                fh.write(f"View {view}:\n")
                for x_px, y_px, lbl in pts:
                    fh.write(f"  {lbl}: x={x_px:.2f}, y={y_px:.2f}\n")
                fh.write("\n")

        all_points[seed] = seed_points

    return all_points


from pathlib import Path
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def plot_views_with_prompts(
    iteration_working_dir=None,
    txt_name = "points.txt",
    img_template = "camera_angle_{idx}.png",
    show = False,
    save_suffix = "_POINTS_UNCROPPED",
):
    """
    Read `points.txt`, load each `camera_angle_{idx}.png`, and overlay the points.

    Parameters
    ----------
    iteration_working_dir : str | Path
        Folder that contains both the images and the text file.
    txt_name : str, default "points.txt"
        Name of the text summary file (your example).
    img_template : str, default "camera_angle_{idx}.png"
        Pattern for image filenames; `{idx}` is replaced with the view index.
    show : bool, default True
        Call `plt.show()` when done.
    save_suffix : str | None, default "_annotated.png"
        If given, write an image copy with this suffix added before the extension
        (e.g. `camera_angle_0_annotated.png`).  Set to `None` to skip saving.

    Returns
    -------
    figs : dict[int, matplotlib.figure.Figure]
        One Matplotlib Figure per view index.
    """
    iteration_working_dir = Path(iteration_working_dir)
    txt_path = iteration_working_dir / txt_name
    if not txt_path.exists():
        raise FileNotFoundError(f"Cannot find {txt_path}")

    # ---------- parse the text ----------
    view_re = re.compile(r"^\s*View\s+(\d+)\s*:")
    pt_re   = re.compile(
        r"^(.*?)\s*:\s*x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)"
    )

    views: dict[int, list[tuple[str, float, float]]] = {}
    current_view = None
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m_view = view_re.match(line)
        if m_view:
            current_view = int(m_view.group(1))
            views[current_view] = []
            continue
        m_pt = pt_re.match(line)
        if m_pt and current_view is not None:
            prompt, x, y = m_pt.group(1), float(m_pt.group(2)), float(m_pt.group(3))
            views[current_view].append((prompt, x, y))
        # silently ignore malformed lines

    if not views:
        raise ValueError("No views / points parsed from the text file.")

    # use the default matplotlib color cycle once, so colors stay consistent
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    figs = {}
    for idx, pts in views.items():
        img_path = iteration_working_dir / img_template.format(idx=idx)
        if not img_path.exists():
            raise FileNotFoundError(f"Cannot find image {img_path}")

        img = mpimg.imread(img_path)

        # ---------- plot ----------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.set_axis_off()

        # overlay points, cycling through default colors
        for i, (_, x, y) in enumerate(pts, start=1):
            color = default_colors[(i - 1) % len(default_colors)]
            ax.scatter(
                x,
                y,
                s=70,
                marker="o",
                edgecolors="k",
                facecolors=color,
                linewidths=1.0,
                zorder=3,
            )

        # build a simple legend (WP‑1, WP‑2, …) with matching colors
        legend_labels = [f"WP-{i}" for i in range(1, len(pts) + 1)]
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                markersize=7,
                linestyle="",
                markerfacecolor=default_colors[(i - 1) % len(default_colors)],
                markeredgecolor="k",
                markeredgewidth=1.0,
            )
            for i in range(1, len(pts) + 1)
        ]
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=8,
        )

        fig.suptitle(f"View {idx}", y=0.98)

        # optional save
        if save_suffix:
            save_name = img_path.with_stem(img_path.stem + save_suffix)
            fig.savefig(save_name, dpi=150, bbox_inches="tight")

        figs[idx] = fig

    if show:
        plt.show()

    return figs

# Example usage:
# all_points = process_and_save_points(config, prompts)



