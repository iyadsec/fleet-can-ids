from pathlib import Path

import pandas as pd


OUT = Path("new_experiments/final_publication_scenarios")


def test_inventory_excludes_model_diversity():
    inv = pd.read_csv(OUT / "audit/source_results_inventory.csv")
    md = inv[inv["experiment"] == "model_diversity"]
    assert not md["eligible_for_final_publication"].any()


def test_campaign_size_200_nodes():
    cs = pd.read_csv(OUT / "results/campaign_size/run_level_metrics.csv")
    fcgnn = cs[cs["method"] == "fcgnn"]
    assert (fcgnn["graph_nodes"] == 200).all()


def test_tables_exist():
    assert (OUT / "tables/table_T1_scenario_definitions.csv").exists()
    assert (OUT / "tables/table_T5_campaign_size_strong.csv").exists()
