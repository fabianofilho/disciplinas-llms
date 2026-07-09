# Disciplinas com LLMs

Dashboard para mapear sobreposicao de conteudo entre disciplinas academicas usando modelos de linguagem (LLMs).

Desenvolvido pelo **LABDAPS** (Laboratorio de Big Data e Analise Preditiva em Saude) — Faculdade de Saude Publica, Universidade de Sao Paulo.

---

## O que e?

Ferramenta que recebe ementas e materiais de disciplinas academicas, usa LLMs para identificar sobreposicao de conteudo entre elas, e exibe o resultado como uma matriz ou rede interativa.

**Casos de uso:**
- Identificar redundancias curriculares em cursos de graduacao e pos-graduacao
- Apoiar reformas curriculares com base em evidencia quantitativa
- Visualizar conexoes interdisciplinares entre areas do conhecimento

---

## Arquitetura

```
Ementas / PDFs
      |
      v
[Backend LLM]  ->  extrai topicos e calcula sobreposicao via prompt
      |
      v
[Banco de dados]  ->  armazena disciplinas, topicos e scores
      |
      v
[Frontend]  ->  visualiza matriz / rede de sobreposicao
```

---

## Roadmap

| Issue | Descricao | Status |
|---|---|---|
| LAB-13 | Definir escopo e wireframe do dashboard | Backlog |
| LAB-14 | Prototipar extracao via LLM (backend) | Backlog |
| LAB-15 | Construir frontend de visualizacao | Backlog |
| LAB-16 | Deploy inicial e validacao com usuarios piloto | Backlog |

---

## Stack prevista

- **Backend:** Python, FastAPI
- **LLM:** Claude (Anthropic API)
- **Frontend:** React ou Streamlit
- **Visualizacao:** D3.js ou Plotly (rede/matriz)
- **Banco:** SQLite ou PostgreSQL

---

## LABDAPS

https://labdaps.fsp.usp.br
