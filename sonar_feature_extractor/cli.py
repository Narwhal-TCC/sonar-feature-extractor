"""
cli.py — Ponto de entrada: sss-feature-extractor

Dois modos de operação
───────────────────────
MODO PIPELINE (recomendado):
  sss-feature-extractor --pipeline pipeline.json --folders ./data/ --output-dir ./out/
  → Lê o JSON, gera um CSV por (sensor_type, model_name).

MODO LEGADO (1 CSV, todos os extractors):
  sss-feature-extractor --folder ./dataset/ --output features.csv

Utilitários:
  sss-feature-extractor --list-extractors
  sss-feature-extractor --list-sensors
  sss-feature-extractor --generate-pipeline example.json
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
from .config import ExtractionConfig
from .folders import resolve_folders, run_multi_folder


def _setup_logging(level: int) -> None:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", level=level, stream=sys.stderr)
    logging.getLogger("skimage").setLevel(logging.ERROR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sss-feature-extractor",
        description="Extrator de features para imagens de Sonar (SSS, FLS...).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODO PIPELINE (recomendado):
  sss-feature-extractor --pipeline pipeline.json \\
      --folders ./data/2010/ ./data/2021/ --output-dir ./outputs/

  sss-feature-extractor --pipeline pipeline.json \\
      --folder ./dataset/ --recursive --output-dir ./outputs/

MODO LEGADO (1 CSV):
  sss-feature-extractor --folder ./dataset/ --output features.csv
  sss-feature-extractor --folders ./2010/ ./2021/ --tag-source --workers 8
  sss-feature-extractor --folder-list pastas.txt --resume

UTILITÁRIOS:
  sss-feature-extractor --list-extractors
  sss-feature-extractor --list-sensors
  sss-feature-extractor --generate-pipeline example.json
""")
    # ── Modo pipeline ──────────────────────────────────────────────
    pl = p.add_argument_group("Modo Pipeline (JSON)")
    pl.add_argument("--pipeline",    metavar="JSON",
                    help="pipeline.json definindo sensors, models e extractors.")
    pl.add_argument("--output-dir",  metavar="DIR", default="./outputs",
                    help="Diretório raiz de saída (default: ./outputs).\n"
                         "Estrutura: {output_dir}/{sensor_type}/{model_name}.csv")
    # ── Entrada ────────────────────────────────────────────────────
    inp = p.add_argument_group("Entrada (ambos os modos)")
    inp.add_argument("--image",       metavar="ARQUIVO")
    inp.add_argument("--label",       metavar="TXT")
    inp.add_argument("--folder",      metavar="DIR")
    inp.add_argument("--folders",     metavar="DIR", nargs="+")
    inp.add_argument("--folder-list", metavar="ARQUIVO")
    inp.add_argument("--recursive",   action="store_true")
    # ── Saída legado ───────────────────────────────────────────────
    leg = p.add_argument_group("Saída — Modo Legado")
    leg.add_argument("--output",     default="sonar_features.csv", metavar="CSV")
    leg.add_argument("--mode",       default="per_object",
                     choices=["per_image","per_object"])
    leg.add_argument("--tag-source", action="store_true")
    # ── Performance ────────────────────────────────────────────────
    perf = p.add_argument_group("Performance")
    perf.add_argument("--workers",          type=int, default=4)
    perf.add_argument("--checkpoint-every", type=int, default=50)
    perf.add_argument("--resume",    action="store_true", default=True)
    perf.add_argument("--no-resume", action="store_true")
    perf.add_argument("--fail-fast", action="store_true")
    # ── Config ─────────────────────────────────────────────────────
    cfg = p.add_argument_group("Configuração")
    cfg.add_argument("--config",      metavar="JSON")
    cfg.add_argument("--save-config", action="store_true")
    # ── Utilitários ────────────────────────────────────────────────
    ut = p.add_argument_group("Utilitários")
    ut.add_argument("--list-extractors",   action="store_true",
                    help="Lista todos os extractors disponíveis.")
    ut.add_argument("--list-sensors",      action="store_true",
                    help="Lista todos os sensor_types registrados.")
    ut.add_argument("--generate-pipeline", metavar="ARQUIVO",
                    help="Gera pipeline.json de exemplo com todos os extractors.")
    # ── Log ────────────────────────────────────────────────────────
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet",   action="store_true")
    return p


# ── Comandos utilitários ──────────────────────────────────────────────────────

def _list_extractors() -> None:
    import sonar_feature_extractor.extractors  # noqa
    from .registry import get_image_registry, get_roi_registry
    img = get_image_registry(); roi = get_roi_registry()
    print("\nExtractors de IMAGEM:")
    for n in sorted(img): print(f"  {n:20s} -{img[n].__class__.__name__}")
    print("\nExtractors de ROI:")
    for n in sorted(roi): print(f"  {n:20s} -{roi[n].__class__.__name__}")
    print(f"\nTotal: {len(img)+len(roi)} extractors\n")


def _list_sensors() -> None:
    import sonar_feature_extractor.sensors  # noqa
    from .sensors import get_available_sensor_types, get_sensor_adapter
    print("\nSensor types disponíveis:")
    for st in sorted(get_available_sensor_types()):
        a = get_sensor_adapter(st)
        print(f"  {st:20s} extensões: {', '.join(a.image_extensions[:4])}")
    print()


def _generate_pipeline(output_path: str) -> None:
    import json
    import sonar_feature_extractor.extractors, sonar_feature_extractor.sensors  # noqa
    from .registry import get_image_registry, get_roi_registry
    from .sensors import get_available_sensor_types

    img_names = sorted(get_image_registry().keys())
    roi_names = sorted(get_roi_registry().keys())
    sensors   = sorted(get_available_sensor_types())

    groups_full = {
        "statistical": [n for n in img_names if n in ("basic_stats","histogram")],
        "texture":     [n for n in img_names if n in ("glcm","haar_wavelet")],
        "spatial":     [n for n in img_names if n in ("spatial_grid","hog")],
        "edges":       [n for n in img_names if n in ("gradient","frequency")],
        "color":       [n for n in img_names if n == "color_channels"],
        "roi":         roi_names,
    }
    groups_min = {
        "core":      [n for n in img_names if n in ("basic_stats","glcm","gradient")],
        "frequency": [n for n in img_names if n in ("frequency","haar_wavelet")],
    }

    pipeline: dict = {
        "_comment": (
            "Gerado por: sss-feature-extractor --generate-pipeline\n"
            "Edite sensors, models e grupos conforme necessário.\n"
            f"Extractors disponíveis: {img_names + roi_names}"
        ),
        "settings": {
            "mode": "per_object", "workers": 4, "output_dir": "./outputs",
            "tag_source": True, "checkpoint_every": 50, "resume": True,
        },
    }
    for st in sensors:
        pipeline[st] = {
            "model_full":       {k: v for k, v in groups_full.items() if v},
            "model_regression": {k: v for k, v in groups_min.items()  if v},
        }

    dest = Path(output_path)
    dest.write_text(json.dumps(pipeline, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Pipeline de exemplo gerado em: {dest.resolve()}")
    print(f"   Sensors  : {sensors}")
    print(f"   Extractors: {img_names + roi_names}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    _setup_logging(level)
    log = logging.getLogger(__name__)

    # ── Utilitários ────────────────────────────────────────────────
    if args.list_extractors:   return _list_extractors()
    if args.list_sensors:      return _list_sensors()
    if args.generate_pipeline: return _generate_pipeline(args.generate_pipeline)

    # ── Configura ExtractionConfig ─────────────────────────────────
    config = ExtractionConfig.from_json(args.config) if args.config else ExtractionConfig()
    config.n_workers        = args.workers
    config.skip_errors      = not args.fail_fast
    config.checkpoint_every = args.checkpoint_every
    resume = not args.no_resume

    if args.save_config:
        cfg_path = Path(args.output).with_suffix(".config.json")
        config.to_json(cfg_path)
        log.info("Config salva em: %s", cfg_path)

    # ════════════════════════════════════════════════════════════════
    # MODO PIPELINE
    # ════════════════════════════════════════════════════════════════
    if args.pipeline:
        from .pipeline_schema import PipelineSpec
        from .engine import PipelineEngine

        if not any([args.folder, args.folders, args.folder_list]):
            parser.error(
                "--pipeline requer ao menos uma fonte de imagens:\n"
                "  --folder DIR | --folders DIR... | --folder-list ARQUIVO"
            )

        spec = PipelineSpec.from_json(args.pipeline)

        # Aplica settings do JSON (CLI tem prioridade)
        s = spec.settings
        if s.mode             is not None: config.mode             = s.mode
        if s.workers          is not None and args.workers == 4:
            config.n_workers = s.workers
        if s.tag_source       is not None: config.tag_source       = s.tag_source
        if s.checkpoint_every is not None and args.checkpoint_every == 50:
            config.checkpoint_every = s.checkpoint_every
        if s.resume           is not None and not args.no_resume:   resume = s.resume

        # CLI --output-dir tem prioridade; JSON aplica só quando CLI está no padrão
        _CLI_OUTPUT_DEFAULT = "./outputs"
        output_dir = Path(args.output_dir if args.output_dir != _CLI_OUTPUT_DEFAULT
                          else (s.output_dir or args.output_dir))
        config.validate()

        resolved = resolve_folders(args.folder, args.folders, args.folder_list, args.recursive)
        log.info("Pipeline: %s | Pastas: %d | Saída: %s",
                 args.pipeline, len(resolved), output_dir)

        results = PipelineEngine(config).run(
            spec=spec, folders=[str(f) for f in resolved],
            output_dir=output_dir, recursive=False, resume=resume,
        )

        W = 58
        print(f"\n{'='*W}")
        print(f"  Pipeline concluido -- {len(results)} CSV(s)")
        print(f"  {'Saida':40s}  {'Linhas':>7}  {'Colunas':>8}")
        print(f"  {'-'*W}")
        for key in sorted(results):
            df = results[key]
            dest = output_dir / (key.replace("/", "/") + ".csv")
            print(f"  {key:40s}  {len(df):>7}  {len(df.columns):>8}")
            print(f"  {'':>42}-> {dest}")
        print(f"{'='*W}\n")
        return

    # ════════════════════════════════════════════════════════════════
    # MODO LEGADO
    # ════════════════════════════════════════════════════════════════
    if args.image:
        import pandas as pd
        from .pipeline import extract_sample
        rows = extract_sample(args.image, args.label, config)
        df   = pd.DataFrame(rows)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        log.info("✅ %d linha(s) → %s", len(rows), args.output)
        return

    if not any([args.folder, args.folders, args.folder_list]):
        parser.error(
            "Informe a fonte de entrada:\n"
            "  --pipeline JSON  (modo multi-sensor)\n"
            "  --image ARQUIVO  (imagem única)\n"
            "  --folder DIR     (pasta)"
        )

    config.mode       = args.mode
    config.tag_source = args.tag_source
    config.validate()
    resolved = resolve_folders(args.folder, args.folders, args.folder_list, args.recursive)
    run_multi_folder([str(f) for f in resolved], args.output, config,
                     recursive=False, resume=resume)


if __name__ == "__main__":
    main()
