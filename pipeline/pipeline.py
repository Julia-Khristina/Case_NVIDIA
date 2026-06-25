import argparse
import json
import sys
import os
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.graph.graph import build_graph
from pipeline.graph.state import AgentState
from pipeline.config.settings import MAX_STARTUPS_PER_RUN

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{message}</cyan>", level="INFO")
logger.add("pipeline.log", rotation="10 MB", level="DEBUG", format="{time} | {level} | {message}")


def carregar_setores() -> list[dict]:
    with open("pipeline/config/setores.json", "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(setor_alvo: str) -> dict:
    logger.info("Iniciando pipeline NVIDIA Startup AI Radar")
    logger.info(f"Setor alvo: {setor_alvo}")

    initial_state: AgentState = {
        "setor_alvo": setor_alvo,
        "consulta_usuario": None,
        "queries_geradas": [],
        "startups_coletadas": [],
        "erros": [],
        "startups_classificadas": [],
        "startups_validadas": [],
        "recomendacoes_rag": [],
        "recomendacoes_finais": [],
        "briefings": [],
        "etapa_atual": "",
        "iteracao": 0
    }

    try:
        graph = build_graph()
        logger.info("Grafo LangGraph compilado com sucesso")

        resultado = graph.invoke(initial_state)
        logger.info("Pipeline executado com sucesso")

        briefings = resultado.get("briefings", [])
        erros = resultado.get("erros", [])
        logger.info(f"Resumo: {len(resultado.get('startups_coletadas', []))} startups coletadas, "
                     f"{len(briefings)} briefings gerados, {len(erros)} erros")

        return resultado

    except Exception as e:
        logger.error(f"Falha na execução do pipeline: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="NVIDIA Startup AI Radar Pipeline")
    parser.add_argument("--setor", type=str, help="ID do setor (ex: healthtech)")
    parser.add_argument("--all-setores", action="store_true", help="Processar todos os setores")
    parser.add_argument("--startup-id", type=str, help="Reprocessar startup específica")
    parser.add_argument("--fase0-only", action="store_true", help="Executar apenas a Fase 0")
    parser.add_argument("--skip-scraping", action="store_true", help="Pular scraping, usar dados do banco")

    args = parser.parse_args()

    if args.fase0_only:
        logger.info("Modo --fase0-only: executando scripts da Fase 0")
        import subprocess
        scripts = ["coletar.py", "enriquecer.py", "limpar.py", "chunking.py", "embeddings.py", "indexar_qdrant.py"]
        for script in scripts:
            caminho = os.path.join("src", script)
            logger.info(f"Executando {caminho}...")
            subprocess.run([sys.executable, caminho], cwd="src")
        logger.info("Fase 0 concluída")
        return

    if args.setor:
        run_pipeline(setor_alvo=args.setor)
        return

    setores = carregar_setores()
    for setor in setores:
        setor_id = setor["id"]
        logger.info(f"Processando setor: {setor['nome']} ({setor_id})")
        try:
            run_pipeline(setor_alvo=setor_id)
        except Exception as e:
            logger.error(f"Erro no setor {setor_id}: {e}")


if __name__ == "__main__":
    main()
