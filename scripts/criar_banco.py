"""
Cria banco SQLite com as disciplinas extraidas pelo scraper.

Uso:
    python scripts/criar_banco.py
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "disciplinas.db"
JSON_PATH = DATA_DIR / "disciplinas.json"


SCHEMA = """
CREATE TABLE IF NOT EXISTS disciplinas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sigla       TEXT UNIQUE,
    nome        TEXT,
    unidade     TEXT,
    area        TEXT,
    nivel       TEXT,
    creditos    TEXT,
    carga_horaria TEXT,
    idioma      TEXT,
    ministrante TEXT,
    ementa      TEXT,
    topicos     TEXT  -- JSON array gerado pelo LLM
);
"""


def criar_banco(disciplinas: list[dict]) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)

    for d in disciplinas:
        conn.execute(
            """INSERT OR IGNORE INTO disciplinas
               (sigla, nome, unidade, area, nivel, creditos, carga_horaria, idioma, ministrante)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                d.get("sigla"),
                d.get("nome"),
                d.get("unidade"),
                d.get("area"),
                d.get("nivel"),
                d.get("creditos"),
                d.get("cargaHoraria"),
                d.get("idioma"),
                d.get("ministrante"),
            ),
        )

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM disciplinas").fetchone()[0]
    conn.close()
    print(f"Banco criado: {DB_PATH} ({total} disciplinas)")


if __name__ == "__main__":
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Rode scrape_disciplinas.py primeiro. Nao encontrado: {JSON_PATH}")
    disciplinas = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    criar_banco(disciplinas)
