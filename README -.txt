# PTW Safety & Sustainability Intelligence (Streamlit + Neo4j)

## 1) Install
python -m pip install streamlit pandas plotly openpyxl neo4j

## 2) Run Streamlit
streamlit run app.py

## 3) Try with sample data
Upload ptw_sample_all_scenarios.csv (or use the sample download in the sidebar).

## 4) Push to Neo4j
- Start Neo4j Desktop DBMS
- In the app, tick 'Enable push to Neo4j' and click 'Push full dataset to Neo4j'

Environment variables (optional):
- NEO4J_URI=bolt://localhost:7687
- NEO4J_USER=neo4j
- NEO4J_PASSWORD=******
- NEO4J_DATABASE=neo4j

## 5) Verify in Neo4j Browser
MATCH (p:Permit) RETURN count(p);
MATCH (p:Permit)-[:CLASHES_WITH]->(q:Permit) RETURN p,q LIMIT 50;

## 6) NeoDash dashboard
Import PTW_Safety_Sustainability_NeoDash.json into NeoDash.
Create filter parameters (names must match):
- status (string array, multi-select)
- risk (string array, multi-select)
- over_budget (boolean or null)
- level2 (boolean or null)
- from (datetime string or null, e.g. 2026-01-01T00:00:00)
- to (datetime string or null)
- permit_no (string or null)

Tip: start with everything empty / null (no filtering):
status=[], risk=[], over_budget=null, level2=null, from=null, to=null, permit_no=null
