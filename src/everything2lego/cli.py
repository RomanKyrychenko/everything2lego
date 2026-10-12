"""CLI entry point.

    everything2lego build MESH.glb --out out/
    everything2lego carve   img1.jpg img2.jpg ... --out out/   # silhouette
    everything2lego triposr img.jpg               --out out/   # neural, single view
    everything2lego hybrid  img1.jpg img2.jpg ... --out out/   # neural + silhouette
    everything2lego demo --out out/

Neural backends require the arm64 venv (see README) and vendor/TripoSR checkout.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import backends, io_bricklink, io_ldraw, io_stl, instructions, pack, pipeline, stability, viewer_export, voxelize

app = typer.Typer(help="Convert a 3D mesh or a set of images of one object into real LEGO bricks.")
console = Console()


def _write_outputs(placements, out_dir: Path, *, model_name: str, report: dict) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    placements = list(placements)
    # One build-step grouping shared by the LDraw file and the PDF.
    steps = instructions.plan_steps(placements)
    ldr = io_ldraw.write_ldr(placements, out_dir / "model.ldr", model_name=model_name, steps=steps)
    xml = io_bricklink.write_wanted_list(placements, out_dir / "wanted.xml")
    view_dir = out_dir / "viewer"
    viewer_index = viewer_export.write_viewer(placements, view_dir, report=report)
    order = instructions._painter_order(placements)
    pdf = instructions.write_pdf(placements, out_dir / "instructions.pdf", order=order, steps=steps)
    preview = out_dir / "preview.png"
    instructions._render_step(placements, set(), (1400, 1000), instructions._bounds(placements),
                              order=order).save(preview)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    stl = io_stl.write_stl(placements, out_dir / "lego.stl")
    return {"ldr": ldr, "bricklink": xml, "viewer": viewer_index, "pdf": pdf,
            "preview": preview, "report": report_path, "lego_stl": stl}


def _print_summary(placements, outputs: dict[str, Path],
                   report: "stability.StabilityReport | None" = None) -> None:
    counts = pack.summarize(placements)
    table = Table(title=f"{len(placements)} parts across {len(counts)} SKUs")
    table.add_column("Part")
    table.add_column("LDraw color", justify="right")
    table.add_column("Count", justify="right")
    for (part_id, color), qty in sorted(counts.items(), key=lambda kv: -kv[1])[:15]:
        table.add_row(part_id, str(color), str(qty))
    if len(counts) > 15:
        table.add_row("...", "", f"({len(counts) - 15} more)")
    console.print(table)

    report = stability.check(placements) if report is None else report
    if report.ok:
        console.print(f"[green]connections:[/green] all {report.total} parts form one supported assembly")
    else:
        console.print(
            f"[yellow]connections:[/yellow] {len(report.floating)} unsupported parts, "
            f"{report.components} assemblies, {len(report.overlaps)} overlaps; review before building."
        )
    if report.weak_connections:
        console.print(f"[yellow]{report.weak_connections} overhangs attach by a single stud; review their strength.[/yellow]")

    console.print()
    for label, path in outputs.items():
        console.print(f"  [green]{label:10s}[/green] {path}")


def _run(grid, out: Path, model_name: str, use_tiles: bool, supports: bool = True,
         use_slopes: bool = True, texture: str = "default",
         color_swaps: dict[int, str] | None = None,
         detect_features: bool = False, color_smoothing: bool = True,
         variety: bool = False, brace_mode: str = "auto",
         architectural_style: bool = False) -> None:
    console.rule("[bold]packing bricks[/bold]")
    try:
        result = pipeline.assemble(grid, use_tiles=use_tiles, use_slopes=use_slopes,
                                    supports=supports, brace_mode=brace_mode)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    placements = result.placements
    if texture and texture != "default":
        before = len(placements)
        placements = pack.substitute_texture(placements, texture)
        swapped = sum(1 for a, b in zip(result.placements, placements) if a.part.ldraw != b.part.ldraw)
        console.print(f"texture '{texture}': swapped {swapped} of {before} parts")
    if color_swaps:
        before_placements = list(placements)
        placements = pack.substitute_by_color(placements, color_swaps)
        swapped = sum(1 for a, b in zip(before_placements, placements) if a.part.ldraw != b.part.ldraw)
        console.print(f"color swaps {color_swaps}: swapped {swapped} parts")
    if detect_features:
        placements, feature_stats = pack.detect_windows_and_doors(placements)
        console.print(f"feature detection: {feature_stats['clusters_found']} clusters found, "
                      f"{feature_stats['windows_placed']} windows + "
                      f"{feature_stats['doors_placed']} doors placed")
    if color_smoothing:
        before_colors = list(placements)
        placements = pack.smooth_isolated_colors(placements)
        smoothed = sum(1 for a, b in zip(before_colors, placements) if a.color_idx != b.color_idx)
        console.print(f"color smoothing: recolored {smoothed} isolated outliers")
    if variety:
        before_variety = list(placements)
        placements = pack.apply_variety(placements)
        swapped = sum(1 for a, b in zip(before_variety, placements) if a.part.ldraw != b.part.ldraw)
        console.print(f"variety: swapped {swapped} bricks to textured alternatives")
    if architectural_style:
        placements, arch_stats = pack.apply_architectural_style(placements)
        console.print(
            f"architectural style: roof line y={arch_stats['roof_line']}; "
            f"foundation {arch_stats['foundation']}, cornice {arch_stats['cornice']}, "
            f"ridge {arch_stats['ridge']}, shingles {arch_stats['shingles']}, "
            f"quoins {arch_stats['quoins']}"
        )
    console.print(f"placed {len(placements)} parts")
    # The passes above may have changed parts and geometry since assemble()
    # checked the structure; report what is actually written.
    check = stability.check(placements)
    report = {**result.report, **pipeline.structure_report(placements, check)}

    console.rule("[bold]writing outputs[/bold]")
    outputs = _write_outputs(placements, out, model_name=model_name, report=report)
    _print_summary(placements, outputs, check)
    if not report["geometrically_connected"]:
        # With --no-supports or a brace mode that leaves parts floating the
        # files are still written, but the model is not buildable as is.
        console.print("[red]The exported model needs structural review; see report.json.[/red]")
        raise typer.Exit(1)


@app.command()
def build(
    mesh: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input .glb/.obj/.stl mesh."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    max_studs: int = typer.Option(32, "--max-studs", min=1, max=128, help="Longer horizontal size in studs."),
    color_samples: int = typer.Option(200_000, "--color-samples",
                                       min=1, help="Surface samples for color transfer."),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles",
                                    help="Replace top-exposed plates with tiles."),
    use_slopes: bool = typer.Option(True, "--slopes/--no-slopes",
                                     help="Swap top-of-column bricks for slopes at 1-brick step edges."),
    supports: bool = typer.Option(True, "--supports/--no-supports", help="Add a foundation and internal bracing."),
    brace_mode: str = typer.Option("auto", "--brace-mode",
                                    help="auto (floors for hollow, columns for solid), floors (internal-floor slabs), columns (legacy vertical), or none (just foundation)."),
    clean_mesh: bool = typer.Option(False, "--clean-mesh/--no-clean-mesh",
                                     help="Merge close vertices, fill holes, Taubin-smooth the input mesh before voxelizing."),
    smooth_iters: int = typer.Option(5, "--smooth-iters", min=0, max=50,
                                      help="Taubin smoothing iterations when --clean-mesh is on."),
    save_mesh: bool = typer.Option(True, "--save-mesh/--no-save-mesh",
                                    help="Also save the input mesh (post-cleanup if enabled) as model.stl and model.glb next to the LEGO outputs."),
    hollow: bool = typer.Option(False, "--hollow/--solid",
                                 help="Carve out the interior of the voxel grid, leaving only an outer shell. Big win for buildings."),
    wall_thickness: int = typer.Option(2, "--wall-thickness", min=1, max=10,
                                        help="Shell thickness in voxels when --hollow is on. 2 = about a 2-stud (16 mm) wall."),
    texture: str = typer.Option("default", "--texture",
                                 help="Post-pack part-for-part swap: log / masonry / curved / smooth / default."),
    windows_color: Optional[str] = typer.Option(None, "--windows-color",
                                                 help="Comma-separated LDraw color codes to swap into 1x2 windows (60592). Example: --windows-color 71,72"),
    doors_color: Optional[str] = typer.Option(None, "--doors-color",
                                               help="LDraw color code(s) to swap into 1x4 doors (60623)."),
    panels_color: Optional[str] = typer.Option(None, "--panels-color",
                                                help="LDraw color code(s) to swap 1x2 bricks into thin 1x2x1 panels (4865)."),
    detect_features: bool = typer.Option(False, "--detect-features/--no-detect-features",
                                          help="After packing, cluster non-wall color regions and swap window/door-shaped clusters for real LEGO window (60592/60594) and door (60623) parts."),
    color_smoothing: bool = typer.Option(True, "--smooth-colors/--no-smooth-colors",
                                          help="Post-pack pass that recolors isolated single-brick color outliers to match the surrounding majority. Kills wall speckling."),
    variety: bool = typer.Option(False, "--variety/--no-variety",
                                  help="Position-hash swap among visually-equivalent wall bricks (basic + log + masonry + headlight + side-stud). Adds visible texture variety."),
    architectural_style: bool = typer.Option(False, "--architectural-style/--no-architectural-style",
                                              help="Apply house-style banding: foundation, cornice, ridge cap, shingle roof, quoin corners. Recolors bricks to make the model read as a real architectural build."),
    color_k: int = typer.Option(5, "--color-k", min=1, max=25,
                                 help="How many nearest surface samples to median-average when color-mode=kdtree."),
    color_mode: str = typer.Option("closest_face", "--color-mode",
                                    help="closest_face (deterministic per-voxel face lookup) or kdtree (random-sample median, legacy)."),
    subvoxels: int = typer.Option(1, "--subvoxels", min=1, max=8,
                                    help="Super-sample the occupancy grid at N times finer resolution, then majority-vote to LEGO scale. Preserves thin walls; costs more RAM/time."),
) -> None:
    """Convert a textured mesh into LEGO."""
    console.rule(f"[bold]loading[/bold] {mesh.name}")
    try:
        m = voxelize.load_mesh(mesh)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"mesh: {len(m.vertices)} verts, {len(m.faces)} faces, bounds {m.bounds.tolist()}")

    if clean_mesh:
        console.rule("[bold]cleaning[/bold] mesh")
        m = voxelize.clean_mesh(m, smooth_iterations=smooth_iters)
        console.print(f"after cleanup: {len(m.vertices)} verts, {len(m.faces)} faces, "
                      f"watertight={m.is_watertight}")

    if save_mesh:
        out.mkdir(parents=True, exist_ok=True)
        try:
            m.export(out / "model.stl")
            m.export(out / "model.glb")
            console.print(f"  [green]stl        [/green] {out / 'model.stl'}")
            console.print(f"  [green]glb        [/green] {out / 'model.glb'}")
        except Exception as exc:
            console.print(f"[yellow]warning: could not save input mesh ({exc})[/yellow]")

    console.rule("[bold]voxelizing[/bold]")
    try:
        grid = voxelize.voxelize(m, max_studs=max_studs, color_samples=color_samples,
                                  hollow=hollow, wall_thickness=wall_thickness,
                                  color_k=color_k, color_mode=color_mode,
                                  subvoxels=subvoxels)
    except (ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz} (y=plates, x=studs, z=studs)")

    color_swaps: dict[int, str] = {}
    def _parse(codes: str | None, target: str):
        if not codes:
            return
        for c in codes.split(","):
            c = c.strip()
            if c:
                color_swaps[int(c)] = target
    _parse(windows_color, "60592.dat")
    _parse(doors_color,   "60623.dat")
    _parse(panels_color,  "4865.dat")

    _run(grid, out, model_name=mesh.stem, use_tiles=use_tiles, use_slopes=use_slopes,
         supports=supports, texture=texture, color_swaps=color_swaps,
         detect_features=detect_features, color_smoothing=color_smoothing,
         variety=variety, brace_mode=brace_mode,
         architectural_style=architectural_style)


@app.command()
def carve(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="Turntable images of one object."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    max_studs: int = typer.Option(32, "--max-studs", min=1, max=128, help="Longer horizontal size in studs."),
    background: Optional[str] = typer.Option(None, "--background",
                                              help="Hex color of the background (e.g. #ffffff). "
                                                   "Auto-detected from image corners if omitted."),
    tolerance: int = typer.Option(28, "--tolerance", min=0, max=255,
                                   help="Per-channel difference from background to count as foreground."),
    min_views_fraction: float = typer.Option(1.0, "--min-views-fraction", min=0.01, max=1,
                                              help="1.0 = strict intersection. 0.9 tolerates one bad view."),
    angles: Optional[str] = typer.Option(None, "--angles",
                                          help="Comma-separated degrees per image. "
                                               "Defaults to evenly-spaced turntable."),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
    masks: Optional[List[Path]] = typer.Option(None, "--mask", exists=True, dir_okay=False,
                                              help="One mask per image, in input order; repeat this option."),
) -> None:
    """Multi-view silhouette space carving (visual hull) -> LEGO."""
    expanded = []
    for path in images:
        if path.is_dir():
            files = [p for p in path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}]
            expanded.extend(sorted(files, key=lambda p: [int(s) if s.isdigit() else s.lower()
                                                        for s in re.split(r"(\d+)", p.name)]))
        else:
            expanded.append(path)
    images = expanded

    console.rule(f"[bold]carving[/bold] {len(images)} views")
    try:
        parsed_angles = [float(a) for a in angles.split(",")] if angles else None
        grid = backends.build(
            "carve", images, max_studs=max_studs, angles=parsed_angles,
            background_hex=background, tolerance=tolerance,
            min_views_fraction=min_views_fraction, masks=masks,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name=images[0].stem, use_tiles=use_tiles, supports=supports)


def _expand_images(images: List[Path]) -> List[Path]:
    """Expand any directories in `images` to their contained image files, in
    natural sort order."""
    expanded: List[Path] = []
    for path in images:
        if path.is_dir():
            files = [p for p in path.iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}]
            expanded.extend(sorted(files, key=lambda p: [int(s) if s.isdigit() else s.lower()
                                                        for s in re.split(r"(\d+)", p.name)]))
        else:
            expanded.append(path)
    return expanded


@app.command()
def triposr(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="Input image(s). Only the first is used for reconstruction."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    max_studs: int = typer.Option(32, "--max-studs", min=1, max=128),
    mc_resolution: int = typer.Option(128, "--mc-resolution", min=32, max=512,
                                       help="Marching-cubes grid resolution for mesh extraction."),
    front_view: int = typer.Option(0, "--front-view", min=0,
                                    help="Index of the image to reconstruct from."),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
) -> None:
    """Neural single-image reconstruction with TripoSR (Stability AI + VAST) -> LEGO."""
    images = _expand_images(images)
    console.rule(f"[bold]TripoSR[/bold] reconstructing from {images[front_view].name}")
    try:
        grid = backends.build(
            "triposr", images, max_studs=max_studs,
            mc_resolution=mc_resolution, front_view=front_view,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name=images[front_view].stem, use_tiles=use_tiles, supports=supports)


@app.command()
def hybrid(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="Turntable images of one object."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    max_studs: int = typer.Option(32, "--max-studs", min=1, max=128),
    mc_resolution: int = typer.Option(128, "--mc-resolution", min=32, max=512),
    front_view: int = typer.Option(0, "--front-view", min=0),
    background: Optional[str] = typer.Option(None, "--background"),
    tolerance: int = typer.Option(28, "--tolerance", min=0, max=255),
    min_views_fraction: float = typer.Option(1.0, "--min-views-fraction", min=0.01, max=1),
    angles: Optional[str] = typer.Option(None, "--angles"),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
) -> None:
    """TripoSR mesh intersected with the multi-view silhouette hull -> LEGO."""
    images = _expand_images(images)
    console.rule(f"[bold]hybrid[/bold] TripoSR + silhouette on {len(images)} views")
    try:
        parsed_angles = [float(a) for a in angles.split(",")] if angles else None
        grid = backends.build(
            "hybrid", images, max_studs=max_studs,
            mc_resolution=mc_resolution, front_view=front_view,
            angles=parsed_angles, background_hex=background,
            tolerance=tolerance, min_views_fraction=min_views_fraction,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name=images[front_view].stem, use_tiles=use_tiles, supports=supports)


@app.command()
def hunyuan(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="Input image(s). Only the first is used for reconstruction."),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    max_studs: int = typer.Option(40, "--max-studs", min=1, max=128),
    steps: int = typer.Option(50, "--steps", min=5, max=200,
                               help="Flow-matching inference steps."),
    resolution: int = typer.Option(384, "--resolution", min=64, max=512,
                                    help="Marching-cubes / octree resolution."),
    guidance: float = typer.Option(5.0, "--guidance", min=0.0, max=20.0),
    front_view: int = typer.Option(0, "--front-view", min=0),
    project_colors: bool = typer.Option(True, "--project-colors/--no-project-colors",
                                         help="Project source photo colors onto the mesh (front-view projection). Off -> monochrome gray."),
    forward_axis: str = typer.Option("-z", "--forward-axis",
                                      help="Camera look direction in mesh space for color projection. One of +x/-x/+y/-y/+z/-z."),
    flip_v: bool = typer.Option(False, "--flip-v/--no-flip-v",
                                 help="Flip vertical when sampling the photo; use if colors end up upside-down."),
    save_mesh: bool = typer.Option(True, "--save-mesh/--no-save-mesh",
                                    help="Also save the raw Hunyuan mesh as model.stl (geometry only) and model.glb (colored)."),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
) -> None:
    """Neural single-image reconstruction with Hunyuan3D-2 (Tencent) -> LEGO.

    Produces a proper textured mesh (not a point cloud), which then goes
    through the same voxelize+pack pipeline as `build`. Best quality of the
    photo->LEGO commands on Apple Silicon.
    """
    images = _expand_images(images)
    console.rule(f"[bold]Hunyuan3D-2[/bold] reconstructing from {images[front_view].name}")
    out.mkdir(parents=True, exist_ok=True)
    stl_path = out / "model.stl" if save_mesh else None
    glb_path = out / "model.glb" if (save_mesh and project_colors) else None
    try:
        grid = backends.build(
            "hunyuan3d", images, max_studs=max_studs,
            num_inference_steps=steps, octree_resolution=resolution,
            guidance_scale=guidance, front_view=front_view,
            project_colors=project_colors, forward_axis=forward_axis,
            flip_v=flip_v,
            save_mesh_path=stl_path, save_colored_mesh_path=glb_path,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name=images[front_view].stem, use_tiles=use_tiles, supports=supports)
    if stl_path and stl_path.exists():
        console.print(f"  [green]mesh (stl)[/green] {stl_path}")
    if glb_path and glb_path.exists():
        console.print(f"  [green]mesh (glb)[/green] {glb_path}")


@app.command("hunyuan21")
def hunyuan21(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="Input image(s). Only the first is used for reconstruction."),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    max_studs: int = typer.Option(40, "--max-studs", min=1, max=128),
    steps: int = typer.Option(50, "--steps", min=5, max=200),
    resolution: int = typer.Option(384, "--resolution", min=64, max=512),
    guidance: float = typer.Option(5.0, "--guidance", min=0.0, max=20.0),
    front_view: int = typer.Option(0, "--front-view", min=0),
    project_colors: bool = typer.Option(True, "--project-colors/--no-project-colors"),
    forward_axis: str = typer.Option("-z", "--forward-axis"),
    flip_v: bool = typer.Option(False, "--flip-v/--no-flip-v"),
    save_mesh: bool = typer.Option(True, "--save-mesh/--no-save-mesh"),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
) -> None:
    """Hunyuan3D-2.1 (newer DiT + updated ShapeVAE) single-image → LEGO.

    Same interface as `hunyuan` but uses the tencent/Hunyuan3D-2.1 model
    with the vendored `hy3dshape` package (see vendor/Hunyuan3D-2.1/).
    Higher-quality geometry than the 2.0 backend.
    """
    images = _expand_images(images)
    console.rule(f"[bold]Hunyuan3D-2.1[/bold] reconstructing from {images[front_view].name}")
    out.mkdir(parents=True, exist_ok=True)
    stl_path = out / "model.stl" if save_mesh else None
    glb_path = out / "model.glb" if (save_mesh and project_colors) else None
    try:
        grid = backends.build(
            "hunyuan3d_v21", images, max_studs=max_studs,
            num_inference_steps=steps, octree_resolution=resolution,
            guidance_scale=guidance, front_view=front_view,
            project_colors=project_colors, forward_axis=forward_axis,
            flip_v=flip_v,
            save_mesh_path=stl_path, save_colored_mesh_path=glb_path,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name=images[front_view].stem, use_tiles=use_tiles, supports=supports)
    if stl_path and stl_path.exists():
        console.print(f"  [green]mesh (stl)[/green] {stl_path}")
    if glb_path and glb_path.exists():
        console.print(f"  [green]mesh (glb)[/green] {glb_path}")


@app.command("hunyuan-mv")
def hunyuan_mv(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="1-4 canonical views (front, back, left, right in this order)."),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    max_studs: int = typer.Option(40, "--max-studs", min=1, max=128),
    steps: int = typer.Option(50, "--steps", min=5, max=200),
    resolution: int = typer.Option(384, "--resolution", min=64, max=512),
    guidance: float = typer.Option(5.0, "--guidance", min=0.0, max=20.0),
    project_colors: bool = typer.Option(True, "--project-colors/--no-project-colors"),
    forward_axis: str = typer.Option("-z", "--forward-axis"),
    flip_v: bool = typer.Option(False, "--flip-v/--no-flip-v"),
    save_mesh: bool = typer.Option(True, "--save-mesh/--no-save-mesh"),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
) -> None:
    """Hunyuan3D-2mv multi-view neural reconstruction (front/back/left/right) -> LEGO.

    Better back-side geometry than the single-image `hunyuan` command, at the
    cost of needing multiple aligned photos of the *same* physical subject.
    """
    images = _expand_images(images)
    console.rule(f"[bold]Hunyuan3D-2mv[/bold] reconstructing from {len(images)} views")
    out.mkdir(parents=True, exist_ok=True)
    stl_path = out / "model.stl" if save_mesh else None
    glb_path = out / "model.glb" if (save_mesh and project_colors) else None
    try:
        grid = backends.build(
            "hunyuan3d_mv", images, max_studs=max_studs,
            num_inference_steps=steps, octree_resolution=resolution,
            guidance_scale=guidance,
            project_colors=project_colors, forward_axis=forward_axis,
            flip_v=flip_v,
            save_mesh_path=stl_path, save_colored_mesh_path=glb_path,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name=images[0].stem, use_tiles=use_tiles, supports=supports)


@app.command()
def photos(
    images: List[Path] = typer.Argument(..., exists=True,
                                         help="Unstructured photos of one subject (any angles)."),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    max_studs: int = typer.Option(40, "--max-studs", min=1, max=128,
                                   help="Longer horizontal size in studs."),
    image_size: int = typer.Option(512, "--image-size",
                                    help="Image resize resolution fed to DUSt3R (224 is faster, 512 more accurate)."),
    niter: int = typer.Option(300, "--niter",
                               help="Global-alignment iterations. 100-200 is often enough."),
    min_conf: float = typer.Option(3.0, "--min-conf",
                                    help="Per-pixel confidence threshold. Lower to keep more points."),
    drop_background: bool = typer.Option(True, "--drop-background/--keep-background",
                                          help="Drop points whose color matches the sampled image-corner colors."),
    background_tolerance: int = typer.Option(30, "--background-tolerance", min=0, max=765,
                                              help="Sum-of-channel distance threshold for background rejection."),
    drop_outliers: bool = typer.Option(True, "--drop-outliers/--keep-outliers",
                                        help="Drop points more than k*MAD from the centroid along any axis."),
    outlier_k: float = typer.Option(3.0, "--outlier-k", min=0.5,
                                     help="MAD multiplier for outlier rejection."),
    snap_ground: bool = typer.Option(True, "--snap-ground/--no-snap-ground",
                                      help="Shift the point cloud so its bottom sits at y=0."),
    ground_percentile: float = typer.Option(2.0, "--ground-percentile", min=0.0, max=50.0,
                                             help="Percentile of y used as the ground level."),
    raw_points: bool = typer.Option(False, "--raw-points",
                                     help="Skip all point-cloud filters (equivalent to disabling each one)."),
    use_tiles: bool = typer.Option(True, "--tiles/--no-tiles"),
    supports: bool = typer.Option(True, "--supports/--no-supports"),
) -> None:
    """Reconstruct from unstructured photos (any angles) using DUSt3R -> LEGO.

    Suited to real-world subjects like a house: pass 4-20 photos taken from
    around the subject, no turntable needed. See the README for photo tips.
    """
    images = _expand_images(images)
    if len(images) < 2:
        raise typer.BadParameter("Provide at least two photos.")
    console.rule(f"[bold]DUSt3R[/bold] reconstructing from {len(images)} photos")
    if raw_points:
        drop_background = drop_outliers = snap_ground = False
    try:
        grid = backends.build(
            "dust3r", images, max_studs=max_studs,
            image_size=image_size, niter=niter, min_conf=min_conf,
            drop_background=drop_background,
            background_tolerance=background_tolerance,
            drop_outliers=drop_outliers, outlier_k=outlier_k,
            snap_ground=snap_ground, ground_percentile=ground_percentile,
        )
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"voxel grid: {grid.shape_yxz}  ({grid.metadata.get('num_points', 0)} points)")
    _run(grid, out, model_name="photos", use_tiles=use_tiles, supports=supports)


@app.command()
def demo(
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    radius: int = typer.Option(6, "--radius", min=1, max=32, help="Sphere radius in studs."),
) -> None:
    """Build a two-tone sphere without needing an input mesh (smoke test)."""
    console.rule("[bold]demo[/bold] two-tone sphere")
    grid = voxelize.dummy_grid_from_primitive("sphere", radius_studs=radius)
    console.print(f"voxel grid: {grid.shape_yxz}")
    _run(grid, out, model_name="demo-sphere", use_tiles=True)


template_app = typer.Typer(help="Parametric architectural templates.")
app.add_typer(template_app, name="template")


@template_app.command("gable-cabin")
def template_gable_cabin(
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    width: int = typer.Option(32, "--width", min=6, max=128),
    depth: int = typer.Option(24, "--depth", min=6, max=128),
    walls: int = typer.Option(5, "--wall-height-bricks", min=2, max=20,
                               help="Wall height in brick heights (each 3 plates)."),
    roof: int = typer.Option(6, "--roof-pitch-bricks", min=1, max=20,
                              help="Roof rise in brick heights."),
    wall_color: int = typer.Option(28, "--wall-color", help="LDraw color code for walls."),
    roof_color: int = typer.Option(72, "--roof-color", help="LDraw color code for roof."),
    foundation_color: int = typer.Option(71, "--foundation-color"),
    quoin_color: int = typer.Option(15, "--quoin-color"),
    door_color: int = typer.Option(6, "--door-color"),
    window_color: int = typer.Option(33, "--window-color"),
    windows_long: int = typer.Option(2, "--windows-per-long-wall", min=0, max=8),
    windows_short: int = typer.Option(1, "--windows-per-short-wall", min=0, max=4),
    supports: bool = typer.Option(False, "--supports/--no-supports"),
) -> None:
    """Parametric gable-roof cabin — no mesh required, ships in seconds."""
    from . import templates as _templates
    from .voxelize import VoxelGrid
    tpl = _templates.GableCabin()
    params = tpl.Params(
        width_studs=width, depth_studs=depth,
        wall_height_bricks=walls, roof_pitch_bricks=roof,
        wall_color_code=wall_color, roof_color_code=roof_color,
        foundation_color_code=foundation_color, quoin_color_code=quoin_color,
        door_color_code=door_color, window_color_code=window_color,
        windows_per_long_wall=windows_long, windows_per_short_wall=windows_short,
    )
    console.rule(f"[bold]template[/bold] {tpl.name}  {width}×{depth}  walls={walls}b  roof={roof}b")
    placements = tpl.build(params)
    console.print(f"template placed {len(placements)} parts (no voxelization)")

    # Write the outputs directly — no pipeline.assemble needed since the
    # template already produced final placements.
    dummy_grid = VoxelGrid(
        palette_idx=__import__("numpy").array([[[-1]]], dtype="int16"),
        metadata={"method": "template", "template": tpl.name},
    )
    outputs = _write_outputs(placements, out, model_name=tpl.name, report={
        "parts": len(placements),
        "template": tpl.name,
        "params": params.__dict__,
    })
    _print_summary(placements, outputs)


@template_app.command("modernist-box")
def template_modernist_box(
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    width: int = typer.Option(40, "--width", min=6, max=128),
    depth: int = typer.Option(20, "--depth", min=6, max=128),
    walls: int = typer.Option(6, "--wall-height-bricks", min=2, max=20),
    parapet: int = typer.Option(1, "--parapet-bricks", min=0, max=6),
    wall_color: int = typer.Option(15, "--wall-color"),
    band_color: int = typer.Option(72, "--band-color"),
    foundation_color: int = typer.Option(0, "--foundation-color"),
    parapet_color: int = typer.Option(72, "--parapet-color"),
    roof_deck_color: int = typer.Option(71, "--roof-deck-color"),
    window_color: int = typer.Option(33, "--window-color"),
    door_color: int = typer.Option(0, "--door-color"),
    windows_long: int = typer.Option(4, "--ribbon-windows-per-long-wall", min=0, max=12),
) -> None:
    """Flat-roof modernist box — parametric International Style."""
    from . import templates as _templates
    from .voxelize import VoxelGrid
    tpl = _templates.ModernistBox()
    params = tpl.Params(
        width_studs=width, depth_studs=depth,
        wall_height_bricks=walls, parapet_bricks=parapet,
        wall_color_code=wall_color, band_color_code=band_color,
        foundation_color_code=foundation_color,
        roof_deck_color_code=roof_deck_color,
        parapet_color_code=parapet_color,
        window_color_code=window_color, door_color_code=door_color,
        ribbon_windows_per_long_wall=windows_long,
    )
    console.rule(f"[bold]template[/bold] {tpl.name}  {width}×{depth}  walls={walls}b")
    placements = tpl.build(params)
    console.print(f"template placed {len(placements)} parts (no voxelization)")
    outputs = _write_outputs(placements, out, model_name=tpl.name, report={
        "parts": len(placements),
        "template": tpl.name,
        "params": params.__dict__,
    })
    _print_summary(placements, outputs)


@app.command()
def optimize(
    mesh: Path = typer.Argument(..., exists=True, help="Path to a .stl/.glb/.ply mesh."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    max_studs: int = typer.Option(28, "--max-studs", min=6, max=64),
    iters: int = typer.Option(20_000, "--iters", min=100),
    t_start: float = typer.Option(0.05, "--t-start"),
    t_end: float = typer.Option(0.0005, "--t-end"),
    w_shape_volume: float = typer.Option(0.7, "--w-shape-volume",
        help="Weight on volume-IoU mismatch (occupancy match)."),
    w_shape_surface: float = typer.Option(1.0, "--w-shape-surface",
        help="Weight on chamfer distance from LEGO surface to *original* mesh surface."),
    w_color: float = typer.Option(0.3, "--w-color"),
    w_diversity: float = typer.Option(0.05, "--w-diversity"),
    w_coherence: float = typer.Option(0.4, "--w-coherence"),
    w_stability: float = typer.Option(0.1, "--w-stability"),
    w_count: float = typer.Option(0.001, "--w-count"),
    seed: int = typer.Option(0, "--seed"),
    color_samples: int = typer.Option(120_000, "--color-samples"),
    texture: Optional[Path] = typer.Option(None, "--texture"),
    slope_prefill: bool = typer.Option(True, "--slope-prefill/--no-slope-prefill",
        help="Greedily pre-place slopes at every slope-ideal surface cell before SA."),
) -> None:
    """Loss-guided LEGO synthesis. Fundamental algorithm — no templates, no
    hand-tuned pipeline.  Minimizes L(A) = shape + color + coherence − diversity
    over cell-grid assemblies via simulated annealing."""
    from . import optimize as opt
    from .optimize.loss import (Weights,
        SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
        SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM,
        SLOPE_CORNER_PP, SLOPE_CORNER_PN, SLOPE_CORNER_NP, SLOPE_CORNER_NN,
        CURVED_XP, CURVED_XM, CURVED_ZP, CURVED_ZM,
        CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM)
    from .optimize.export import assembly_to_placements

    console.rule("[bold]optimize[/bold]  loss-guided LEGO synthesis")
    console.print(f"  mesh: {mesh}  max_studs={max_studs}  iters={iters}")

    target = opt.build_target(mesh, max_studs=max_studs,
                                color_samples=color_samples)
    console.print(f"  target: {target.shape}  "
                  f"occupied={int(target.occupied.sum())}  "
                  f"surface={int(target.surface.sum())}")

    weights = Weights(shape_volume=w_shape_volume,
                       shape_surface=w_shape_surface,
                       color=w_color, diversity=w_diversity,
                       coherence=w_coherence, stability=w_stability, count=w_count)

    def on_progress(step: int, weighted):
        console.print(f"    [dim]step {step:>6}[/dim]  {weighted.as_line()}")

    initial = opt.initial_assembly_with_slopes(target) if slope_prefill else None
    if initial is not None:
        # Count how many cells the pre-fill covered so the user can see the
        # jump from "all plain" to "slope-covered surface".
        import numpy as np
        n_pref = int(np.sum(np.isin(initial.cell_type,
                                      [SLOPE_XP, SLOPE_XM, SLOPE_ZP, SLOPE_ZM,
                                       SLOPE_INV_XP, SLOPE_INV_XM, SLOPE_INV_ZP, SLOPE_INV_ZM,
                                       SLOPE_CORNER_PP, SLOPE_CORNER_PN,
                                       SLOPE_CORNER_NP, SLOPE_CORNER_NN,
                                       CURVED_XP, CURVED_XM, CURVED_ZP, CURVED_ZM,
                                       CHEESE_XP, CHEESE_XM, CHEESE_ZP, CHEESE_ZM])))
        console.print(f"  slope-prefill: {n_pref} slope-family cells placed")

    result = opt.optimize(target, weights=weights, iters=iters,
                            T_start=t_start, T_end=t_end,
                            seed=seed, on_progress=on_progress,
                            initial=initial)

    console.print(f"  init  L={result.initial_loss.total:.4f}")
    console.print(f"  best  L={result.best_loss.total:.4f}")
    console.print(f"  accepts={result.n_accepts}/{result.n_proposals}  "
                    f"elapsed={result.elapsed_s:.1f}s")

    placements = assembly_to_placements(result.best)
    console.print(f"  emitted {len(placements)} parts")

    # Also save the source mesh as source.stl next to the LEGO output so the
    # user can compare "what the loss was targeting" vs "what the LEGO built".
    # `voxelize.load_mesh` normalises Scene → Trimesh already.
    out.mkdir(parents=True, exist_ok=True)
    source_mesh = voxelize.load_mesh(mesh)
    source_mesh.export(out / "source.stl")
    console.print(f"  saved source mesh → {out / 'source.stl'}")

    outputs = _write_outputs(placements, out, model_name="optimized", report={
        "parts": len(placements),
        "iterations": iters,
        "initial_loss": result.initial_loss.total,
        "best_loss": result.best_loss.total,
        "best_raw": {
            "shape_volume": result.best_raw.shape_volume,
            "shape_surface": result.best_raw.shape_surface,
            "color": result.best_raw.color,
            "diversity": result.best_raw.diversity,
            "coherence": result.best_raw.coherence,
            "stability": result.best_raw.stability,
            "count": result.best_raw.count,
        },
        "accepts": result.n_accepts,
        "proposals": result.n_proposals,
        "elapsed_s": result.elapsed_s,
        "weights": weights.__dict__,
    })
    _print_summary(placements, outputs)


@app.command()
def semantic(
    mesh: Path = typer.Argument(..., exists=True, help="Path to .stl/.glb/.ply mesh."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
    max_studs: int = typer.Option(32, "--max-studs", min=6, max=128),
    n_samples: int = typer.Option(100_000, "--n-samples",
        help="Surface point samples for the RegionDetector."),
    subvoxels: int = typer.Option(1, "--subvoxels", min=1, max=4,
        help="Super-sample the occupancy grid at N× finer resolution and "
              "majority-vote back to LEGO scale.  N=2 preserves thin walls "
              "and window frames; N=3-4 for very fine trim."),
    hollow: bool = typer.Option(False, "--hollow/--no-hollow",
        help="Hollow the interior, leaving only a structural skeleton "
              "(shell + floors + columns + foundation).  Cuts interior part "
              "count by 70-80%% while keeping the model buildable."),
    shell_thickness: int = typer.Option(2, "--shell-thickness", min=1, max=6),
    floor_spacing: int = typer.Option(12, "--floor-spacing", min=1, max=32),
    column_spacing: int = typer.Option(8, "--column-spacing", min=1, max=32),
    milp_time: float = typer.Option(5.0, "--milp-time",
        help="Per-region MILP time budget (seconds); greedy fallback if exceeded."),
    diversity_weight: float = typer.Option(0.02, "--diversity-weight"),
    refine: bool = typer.Option(False, "--refine/--no-refine",
        help="Enable v3 loss-guided refinement after the semantic-MILP init."),
    refine_iters: int = typer.Option(3000, "--refine-iters"),
    grad_refine: bool = typer.Option(False, "--grad-refine/--no-grad-refine",
        help="Enable v3 hybrid gradient refinement AFTER SA."),
    grad_iters: int = typer.Option(15, "--grad-iters"),
    soft_train: bool = typer.Option(False, "--soft-train/--no-soft-train",
        help="Enable v4 soft (Gumbel-Softmax) training pass — replaces SA "
             "with true gradient descent."),
    soft_iters: int = typer.Option(300, "--soft-iters"),
    soft_lr:    float = typer.Option(0.3, "--soft-lr"),
    photos: list[Path] = typer.Option(None, "--photos",
        help="Zero or more photo files. Each fed as a photo-loss target during "
             "--soft-train.  First photo becomes the +Z (front) view, second is "
             "+X (side), third is +Y (top)."),
    w_photo_pixel:  float = typer.Option(1.0, "--w-photo-pixel",
        help="Weight of the per-pixel photo loss; only used with --photos."),
    w_photo_percep: float = typer.Option(0.0, "--w-photo-percep",
        help="Weight of the perceptual (VGG) photo loss; only used with --photos."),
    w_shape_surface: float = typer.Option(1.0, "--w-shape-surface"),
    w_shape_volume: float = typer.Option(2.0, "--w-shape-volume"),
    w_color: float = typer.Option(1.0, "--w-color"),
    w_diversity: float = typer.Option(0.0, "--w-diversity"),
    w_coherence: float = typer.Option(0.02, "--w-coherence"),
    w_count: float = typer.Option(0.001, "--w-count"),
) -> None:
    """Semantic-MILP pipeline (v2). Segments the mesh into typed regions
    and packs each with an ILP for provably near-optimal parts choice.
    See ARCHITECTURE_V2.md for the design."""
    from .semantic_v2 import SemanticLegoPipeline, PipelineConfig
    from .semantic_v2.detector import DetectorConfig
    from .semantic_v2.builders import BuilderConfig
    from .semantic_v2.interior import InteriorConfig
    cfg = PipelineConfig(
        max_studs=max_studs,
        n_samples=n_samples,
        color_samples=n_samples,
        subvoxels=subvoxels,
        detector=DetectorConfig(),
        builder=BuilderConfig(
            milp_time_limit_s=milp_time,
            diversity_weight=diversity_weight,
        ),
        interior=InteriorConfig(
            hollow=hollow,
            shell_thickness=shell_thickness,
            floor_spacing=floor_spacing,
            column_spacing=column_spacing,
        ),
    )
    console.rule("[bold]semantic[/bold]  v2 pipeline (RANSAC regions + MILP)")
    console.print(f"  mesh: {mesh}  max_studs={max_studs}  n_samples={n_samples}")
    pipeline = SemanticLegoPipeline(cfg)
    result = pipeline.run(mesh)
    console.print(f"  regions detected: {len(result.regions)}")
    for reg in result.regions:
        console.print(f"    - {reg.label.value}  n_points={reg.n_points()}")
    console.print(f"  per-region parts:")
    for k, v in sorted(result.per_region_counts.items(),
                         key=lambda kv: -kv[1]):
        console.print(f"    {k:>40s}  {v}")
    console.print(f"  total placements: {len(result.placements)}")

    final_placements = result.placements
    refine_report: dict = {}
    if refine and result.placements:
        from .semantic_v2 import (LossFunction, LossWeights, Refiner,
                                    build_move_pool)
        from . import colors as _colors
        console.rule("[bold]refine[/bold]  v3 loss-guided refinement")
        target = opt_build_target_for_refine(mesh, max_studs, n_samples, subvoxels)
        loss = LossFunction(target, LossWeights(
            shape_surface=w_shape_surface, shape_volume=w_shape_volume,
            color=w_color, diversity=w_diversity,
            coherence=w_coherence, count=w_count,
        ))
        # Colour pool = all palette indices (Recolor's target_lookup picks
        # the nearest palette to the mesh's true colour at that point).
        pal = _colors.palette()
        color_pool = list(range(len(pal)))

        def _color_lookup(pl):
            # Fast: use LossFunction's cached KDTree to find the nearest
            # mesh sample; return the nearest palette to its true colour.
            import numpy as np
            Tinv = np.linalg.inv(target.mesh_to_voxel)
            cx = pl.x_stud + pl.w_world / 2
            cy = pl.y_plate + pl.part.height_units / 2
            cz = pl.z_stud + pl.d_world / 2
            world = (Tinv @ np.array([cx, cy, cz, 1.0], dtype=np.float32))[:3]
            _, idx = loss._mesh_kd.query(world)
            rgb = target.surface_colors[idx].astype(np.int32)
            pal_rgb = loss._palette_rgb
            d = np.linalg.norm(pal_rgb - rgb, axis=1)
            return int(d.argmin())

        # Compute mesh occupancy + surface mask for TrimInteriorMove.
        import numpy as np
        mesh_occ = target.voxel_grid.occupied()
        surf = np.zeros_like(mesh_occ)
        surf[:-1, :, :] |= mesh_occ[:-1, :, :] & ~mesh_occ[1:, :, :]
        surf[1:, :, :]  |= mesh_occ[1:, :, :]  & ~mesh_occ[:-1, :, :]
        surf[:, :-1, :] |= mesh_occ[:, :-1, :] & ~mesh_occ[:, 1:, :]
        surf[:, 1:, :]  |= mesh_occ[:, 1:, :]  & ~mesh_occ[:, :-1, :]
        surf[:, :, :-1] |= mesh_occ[:, :, :-1] & ~mesh_occ[:, :, 1:]
        surf[:, :, 1:]  |= mesh_occ[:, :, 1:]  & ~mesh_occ[:, :, :-1]
        moves = build_move_pool(_color_lookup, color_pool,
                                    mesh_occ=mesh_occ, surface_mask=surf)
        def on_progress(step, weighted):
            console.print(f"    step {step:>5}  {weighted.as_line()}")
        refiner = Refiner(loss=loss, moves=moves, iters=refine_iters,
                            on_progress=on_progress, progress_every=refine_iters // 10)
        r = refiner.run(result.placements)
        console.print(f"  init  {r.initial_weighted.as_line()}")
        console.print(f"  best  {r.best_weighted.as_line()}")
        console.print(f"  accepts {r.n_accepted}/{r.n_proposed}  "
                        f"elapsed {r.elapsed_s:.1f}s  parts {len(r.best)}")
        final_placements = r.best
        refine_report = {
            "initial_loss": r.initial_weighted.total,
            "best_loss": r.best_weighted.total,
            "n_accepted": r.n_accepted, "n_proposed": r.n_proposed,
            "elapsed_s": r.elapsed_s, "n_initial_parts": len(result.placements),
            "n_final_parts": len(r.best),
        }

    if grad_refine and final_placements:
        from .semantic_v2 import (LossFunction, LossWeights, GradientRefiner,
                                    MeshTarget)
        console.rule("[bold]grad-refine[/bold]  hybrid gradient / steepest-descent")
        target = opt_build_target_for_refine(mesh, max_studs, n_samples, subvoxels)
        loss = LossFunction(target, LossWeights(
            shape_surface=w_shape_surface, shape_volume=w_shape_volume,
            color=w_color, diversity=w_diversity,
            coherence=w_coherence, count=w_count,
        ))
        def gr_progress(step, weighted):
            console.print(f"    outer {step:>3}  {weighted.as_line()}")
        gr = GradientRefiner(loss=loss, target=target,
                                outer_iters=grad_iters,
                                on_progress=gr_progress)
        gres = gr.run(final_placements)
        console.print(f"  init grad-refine  {gres.initial_weighted.as_line()}")
        console.print(f"  best grad-refine  {gres.best_weighted.as_line()}")
        console.print(f"  recolor steps={gres.n_recolor_steps}  "
                        f"descent steps={gres.n_descent_steps}  "
                        f"descent accepts={gres.n_descent_accepts}  "
                        f"elapsed={gres.elapsed_s:.1f}s  parts={len(gres.best)}")
        final_placements = gres.best
        refine_report["grad_initial_loss"] = gres.initial_weighted.total
        refine_report["grad_best_loss"] = gres.best_weighted.total
        refine_report["grad_elapsed_s"] = gres.elapsed_s
        refine_report["grad_descent_accepts"] = gres.n_descent_accepts

    if soft_train and result.placements:
        try:
            import torch
            import numpy as np
        except ImportError as exc:
            raise typer.BadParameter(
                "--soft-train needs PyTorch (pip install torch); run without it "
                "or install torch") from exc
        else:
            from .semantic_v2 import (SoftAssembly, SoftLoss, SoftLossWeights,
                                        SoftTrainer, SoftRenderer, MeshTarget)
            console.rule("[bold]soft-train[/bold]  v4 Gumbel-Softmax + Adam")
            target = opt_build_target_for_refine(mesh, max_studs, n_samples, subvoxels)
            from . import colors as _colors
            n_col = len(_colors.palette())
            occ = target.voxel_grid.occupied()
            pi = target.voxel_grid.palette_idx
            # Apply the same hollowing to the soft-train target so the
            # trainer's shape_volume MSE doesn't fight the empty
            # interior.  Keeps shell + floors + columns + foundation
            # as OCCUPIED cells; everything else becomes EMPTY.
            if hollow:
                from .semantic_v2.interior import _hollow_with_supports
                keep = _hollow_with_supports(
                    remaining=occ.copy(), occ=occ,
                    shell_thickness=shell_thickness,
                    floor_spacing=floor_spacing,
                    column_spacing=column_spacing,
                    foundation_plates=3,
                )
                occ = keep
                # Interior palette indices go to structural gray.
                import numpy as np
                pi = np.where(keep, pi, -1)
            # Build the geometric type prior once and use it for BOTH the
            # SoftLoss's KL term and the SoftAssembly's initial state so
            # the trainer starts in a good place (slopes at edges, tiles
            # at top, plates in interior — not random or all-plate).
            from .semantic_v2.soft_loss import _build_geometric_type_prior
            # Feed surface points + normals so the type prior can pick
            # slope directions from actual mesh pitches, not from the
            # empty-neighbour heuristic (fewer stair-step artefacts).
            sp_vox = None
            sn = None
            if getattr(target, "surface_points", None) is not None \
                    and getattr(target, "mesh_to_voxel", None) is not None:
                pts_h = np.hstack([target.surface_points,
                                     np.ones((len(target.surface_points), 1),
                                                dtype=np.float32)])
                sp_vox = (pts_h @ target.mesh_to_voxel.T)[:, :3].astype(np.float32)
                sn = target.surface_normals.astype(np.float32)
            geo_prior = _build_geometric_type_prior(
                occ, surface_points_vox=sp_vox, surface_normals=sn)
            # Palette lock: only allow LDraw colours that actually appear
            # in the mesh's voxel target.  Prevents trainer drift into
            # off-tone entries (e.g., Rust bleeding onto a slate roof).
            allowed_palette = np.unique(pi[occ])
            allowed_palette = allowed_palette[allowed_palette >= 0]
            # Init the SoftAssembly from the semantic-MILP placements
            # (idea #1): those already carry per-region packing
            # decisions from PlanarWallBuilder / PlanarRoofBuilder etc.
            # The soft trainer can then REFINE colours + fix gaps
            # without having to rediscover the region structure.
            #
            # Per-region palette lock (idea #2): each region's cells are
            # restricted to the palette indices that appear in THAT
            # region's own voxel target, not the mesh-wide palette.
            per_region_palette: dict = {}
            if result.regions and result.placements:
                # For each Region, build a boolean cell-mask by
                # rasterising its bounds into voxel space.
                from .semantic_v2 import _build_region_cell_masks
                per_region_palette = _build_region_cell_masks(
                    target, result.regions, pi, occ)
            sa = SoftAssembly.from_placements(
                occ.shape, final_placements, n_col,
                geometric_prior=geo_prior,
                allowed_palette=allowed_palette,
                per_region_palette=per_region_palette or None)
            # Photo targets.
            photo_views = {}
            renderer = None
            perceptual = None
            if photos:
                from PIL import Image
                renderer = SoftRenderer(n_col)
                axis_order = ["z", "x", "y"]
                for i, path in enumerate(photos[:3]):
                    im = Image.open(path).convert("RGB")
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    photo_views[axis_order[i]] = torch.from_numpy(arr)
                if w_photo_percep > 0:
                    from .semantic_v2.perceptual import PerceptualLoss
                    perceptual = PerceptualLoss()
            sl = SoftLoss(occ, pi, n_col,
                            SoftLossWeights(
                                shape_volume=w_shape_volume,
                                color=w_color,
                                coherence=w_coherence,
                                diversity=w_diversity,
                                photo_pixel=w_photo_pixel if photos else 0.0,
                                photo_percep=w_photo_percep if photos and perceptual else 0.0,
                            ),
                            photo_views=photo_views, renderer=renderer,
                            perceptual=perceptual)
            def st_prog(step, brk):
                console.print(f"    step {step:>4}  {brk.as_line()}")
            trainer = SoftTrainer(soft_loss=sl, lr=soft_lr, iters=soft_iters,
                                     tau_start=1.0, tau_end=0.05,
                                     on_progress=st_prog,
                                     progress_every=max(1, soft_iters // 8))
            sres = trainer.run(sa)
            console.print(f"  init soft loss: {sres.initial_loss:.4f}")
            console.print(f"  best soft loss: {sres.best_loss:.4f}")
            console.print(f"  elapsed: {sres.elapsed_s:.1f}s  parts: {len(sres.placements)}")
            final_placements = sres.placements
            refine_report["soft_initial_loss"] = sres.initial_loss
            refine_report["soft_best_loss"] = sres.best_loss
            refine_report["soft_elapsed_s"] = sres.elapsed_s
            refine_report["soft_n_parts"] = len(sres.placements)

    if final_placements:
        check = stability.check(final_placements)
        outputs = _write_outputs(final_placements, out,
                                  model_name="semantic",
                                  report={**pipeline.structure_report(final_placements, check),
                                          "regions": len(result.regions),
                                          "per_region": result.per_region_counts,
                                          "refine": refine_report})
        _print_summary(final_placements, outputs, check)
        if not check.ok:
            console.print("[red]The exported model needs structural review; see report.json.[/red]")
            raise typer.Exit(1)
    else:
        console.print("[yellow]no placements produced — check detector output[/yellow]")


def opt_build_target_for_refine(mesh_path, max_studs, n_samples, subvoxels=1):
    """Rebuild the MeshTarget with the same settings the pipeline used, so
    the Refiner's loss agrees on coordinate frame."""
    from .semantic_v2 import MeshTarget
    return MeshTarget.from_path(mesh_path, max_studs=max_studs,
                                  n_samples=n_samples, color_samples=n_samples,
                                  subvoxels=subvoxels)


if __name__ == "__main__":
    app()
