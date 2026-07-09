"""
Scraper de disciplinas do Janus USP.

Usa Playwright para renderizar o JSF/AJAX e extrair dados via DWR.
Salva em data/disciplinas.json e data/disciplinas.csv.

Uso:
    pip install playwright
    playwright install chromium
    python scripts/scrape_disciplinas.py
"""
from __future__ import annotations

import json
import csv
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, Page

BASE_URL = "https://uspdigital.usp.br/janus"
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def interceptar_dados_dwr(page: Page) -> list[dict]:
    """Captura respostas DWR via interceptacao de rede."""
    dados_capturados: list[dict] = []

    def on_response(response):
        if "dwr" in response.url and response.status == 200:
            try:
                body = response.body().decode("latin-1", errors="ignore")
                if "sigla" in body.lower() or "disciplina" in body.lower():
                    dados_capturados.append({"url": response.url, "body": body[:5000]})
            except Exception:
                pass

    page.on("response", on_response)
    return dados_capturados


def parsear_dwr_response(body: str) -> list[dict]:
    """Extrai lista de disciplinas da resposta DWR."""
    disciplinas = []

    # DWR retorna objetos JS como: s0.sigla="FSP5710"; s0.nome="Epidemiologia"; etc.
    blocos = re.split(r"var s\d+=", body)
    for bloco in blocos[1:]:
        obj: dict = {}
        for campo in ["sigla", "nome", "unidade", "area", "creditos",
                       "cargaHoraria", "nivel", "idioma", "ministrante"]:
            m = re.search(rf'{campo}="([^"]*)"', bloco)
            if m:
                obj[campo] = m.group(1)
        if obj.get("sigla") or obj.get("nome"):
            disciplinas.append(obj)

    return disciplinas


def scrape_disciplinas_oferecidas() -> list[dict]:
    """Navega pelo Janus e extrai disciplinas oferecidas."""
    disciplinas: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        capturadas = interceptar_dados_dwr(page)

        print("Acessando pagina de disciplinas oferecidas...")
        page.goto(f"{BASE_URL}/componente/disciplinasOferecidasInicial.jsf",
                  wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # Tenta acionar listagem publica (sem login)
        try:
            page.click("text=Listar", timeout=5000)
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        # Coleta dados capturados via DWR
        for item in capturadas:
            disciplinas.extend(parsear_dwr_response(item["body"]))

        # Fallback: extrai da tabela HTML se houver
        rows = page.query_selector_all("table tr")
        for row in rows[1:]:  # pula cabecalho
            cells = row.query_selector_all("td")
            if len(cells) >= 2:
                sigla = cells[0].inner_text().strip()
                nome = cells[1].inner_text().strip()
                if sigla and nome and re.match(r"[A-Z]{2,4}\d{4}", sigla):
                    disciplinas.append({"sigla": sigla, "nome": nome})

        browser.close()

    return disciplinas


def salvar(disciplinas: list[dict]) -> None:
    if not disciplinas:
        print("Nenhuma disciplina extraida.")
        return

    # JSON
    json_path = OUTPUT_DIR / "disciplinas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(disciplinas, f, ensure_ascii=False, indent=2)
    print(f"Salvo: {json_path} ({len(disciplinas)} registros)")

    # CSV
    csv_path = OUTPUT_DIR / "disciplinas.csv"
    campos = list({k for d in disciplinas for k in d})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(disciplinas)
    print(f"Salvo: {csv_path}")


if __name__ == "__main__":
    disciplinas = scrape_disciplinas_oferecidas()
    salvar(disciplinas)
