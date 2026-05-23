# CLAUDE.md — sonar-feature-extractor

> Documentação detalhada: ver [`agents/CLAUDE.md`](ai-helpers/agents/CLAUDE.md) na raiz do projeto. Este arquivo é um pointer leve.

## Visão Geral

`sonar-feature-extractor` (v3.1.0) é um pipeline Python CLI que transforma imagens de sonar subaquático (SSS, FLS) + anotações em CSVs numéricos para treinar modelos de ML. Suporta múltiplos sensores (registry pattern com `@register_sensor` e `@register_image`) e gera CSVs por modelo a partir de uma config JSON única.

## Documentação completa

- **[`agents/CLAUDE.md`](agents/CLAUDE.md)** — guia técnico completo: layout do pacote, padrões de extensão, arquitetura, datasets, extractors, constraints Windows.
- **[`docs/handoff_document.md`](docs/handoff_document.md)** — documento de handoff técnico com estado v3.1.0 + roadmap.
- **[`docs/extractors_info.md`](docs/extractors_info.md)** — referência detalhada dos extractors.
- **[`docs/aws-architecture.md`](docs/aws-architecture.md)** — arquitetura cloud AWS: EC2 + S3 + Jupyter, deploy, uso e troubleshooting.
- **[`docs/CHANGELOG-aws-integration.md`](docs/CHANGELOG-aws-integration.md)** — resumo técnico de todas as alterações da integração cloud.

## Regras Obrigatórias do Projeto

> **DIRETIVAS**: regras que se aplicam a TODA sessão, TODO agente, TODA interação neste projeto.

### R1 — Integração Obsidian automática
Vault: `C:\Users\jooju\OneDrive\Documentos\Obsidian Vault`. MCP `obsidian-vault` para leitura (Mode B). Para writes, use o `Write` tool diretamente no caminho do vault. **No início de sessões com trabalho significativo**, consulte `memory.md` na raiz do vault para contexto. Skill: `/integrate-obsidian`.

### R2 — Manter o vault atualizado
Após qualquer trabalho relevante (features, fixes, arquitetura, TODOs, decisões), **atualize** as notas:
- `04-projects/sonar-feature-extractor/` — arquitetura, extractors, trabalho pendente
- `04-projects/aws-infra/` — infraestrutura cloud
- `05-experiments/EXP-YYYYMMDD-NNN.md` — experimentos ML
- `02-wiki/` — conceitos técnicos novos
- `memory.md` (raiz do vault) — estado, decisões, próximos passos
Skill: `/update-vault`.

### R3 — Manter diagramas de arquitetura atualizados (Mermaid)
Quando a arquitetura mudar, **atualize** os diagramas Mermaid:
- `04-projects/sonar-feature-extractor/architecture.md` — pipeline
- `04-projects/aws-infra/overview.md` — cloud

### R4 — Nunca `git push`
Apenas `git commit`. **Nunca** `git push`. O usuário revisa e faz push manualmente.

## Constraints Críticas (Windows)

1. **`pip install .`** (não `pip install -e .`) — caminho com acentos (`Área de Trabalho`) quebra o `.pth` editável.
2. **`cv2.imdecode` + `read_bytes`** — nunca `cv2.imread(str(path))` direto; falha silenciosa com Unicode.
3. **Workers picklavéis** — `ProcessPoolExecutor` no Windows usa `spawn`. Sempre `functools.partial` com funções de `_worker.py`. **Nunca** closures locais.

## Próximos passos pendentes

- **Alta prioridade:** sensor `fls_uxo_aris` (Dahn et al. 2024). Criar `sensors/fls_dahn.py` + `extractors/fls_pose_meta.py` + `docs/fls_dahn2024.md`. Antes, examinar https://github.com/dfki-ric/uxo-dataset2024 para entender o formato de pose 6-DOF.
- **Média:** suite pytest (`tests/test_*.py`) com fixtures de imagens sintéticas.
- **Média:** suporte multi-sensor no mesmo `pipeline.json` (já funciona teoricamente, falta teste end-to-end).

## Comandos Quick Reference

```bash
# Local
sonar-feature-extractor --list-extractors
sonar-feature-extractor --list-sensors
sonar-feature-extractor --pipeline tests/pipeline_full.json --folder ./data/sss/2010/ --output-dir ./outputs/

# Cloud (EC2)
ssh -i ~/.aws/narwhal-keypair.pem ec2-user@<IP>
bash ~/run-pipeline-s3.sh                    # sync S3 → pipeline → sync back
# JupyterLab: http://<IP>:8888/?token=narwhal-jupyter-2024
```

## Infraestrutura Cloud

- **EC2:** `i-0dc8730e5b1edf020` (t3.medium, Amazon Linux 2023) — **parada por padrão**
- **S3:** `narwhal-data-293379721401` (raw/, features/, pipelines/, notebooks/)
- **Terraform:** projeto `data-ingestion/terraform/` (S3 backend em `narwhal-state-293379721401`)
- **Iniciar:** `aws ec2 start-instances --instance-ids i-0dc8730e5b1edf020`
- **IP muda** a cada start — verificar com `aws ec2 describe-instances`
