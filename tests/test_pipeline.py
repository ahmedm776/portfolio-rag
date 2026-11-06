"""v2.5 pipeline behaviour: attribution, scope guard, wrapped bullets, historical labelling."""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retriever import Retriever
from pipeline import Pipeline
from extractive import sentences

P = Pipeline(Retriever())

def test_scope_guard_unknown_term():
    for q in ["What accuracy did the NumPy network get on Fashion-MNIST?",
              "What is the latency of the GTSRB model on a Raspberry Pi?"]:
        assert P.ask(q)["abstained"], q

def test_metric_not_in_routed_doc():
    assert P.ask("What is the F1 score of the loan approval prediction model?")["abstained"]

def test_project_attribution():
    a = P.ask("What accuracy did the GTSRB traffic-sign model reach on the test split?")
    assert not a["abstained"] and all(c.startswith("traffic-sign-gtsrb") for c in a["citations"])
    assert "98.89" in a["answer"]

def test_answer_never_joins_documents():
    for q in ["What is the baseline CNN accuracy on CIFAR-10?", "How many loan records are there?",
              "How many rows were quarantined in the telecom data?"]:
        a = P.ask(q)
        assert len({c.split("#")[0] for c in a["citations"]}) <= 1

def test_wrapped_bullet_is_one_sentence():
    text = "T - S\n- The original README was never recovered; the two-person\n  coursework framing is unconfirmed.\n- Second item here is long enough to count."
    sents = sentences(text)
    assert any("two-person coursework framing is unconfirmed" in s for s in sents)

def test_current_preferred_over_historical():
    # Current headline is the 2026-11-06 continuation candidate (89.51); 87.55 is the
    # superseded 20-epoch baseline kept for provenance.
    a = P.ask("What is the ResNet accuracy on the official CIFAR-10 test split?")
    assert "89.51" in a["answer"] and not a["answer"].startswith("(historical)")
    h = P.ask("What was the ResNet accuracy in the custom assignment split?")
    assert "0.852" in h["answer"] or "historical" in h["answer"].lower()

# Development regression tests: these use KNOWN failures from the exposed final
# set and probe list (both are development data, not validation).
def test_dev_known_failures_fixed():
    assert "6,967" in P.ask("How many movies rows are in the pedagogical branch after IQR exclusion?")["answer"]
    assert "8,900" in P.ask("How many TXT files did the BBC download include?")["answer"]
    assert "21277" in P.ask("How many terms did the caption search vocabulary have in the original notebook run?")["answer"].replace(",", "")
    assert "0.67" in P.ask("How many parameters does the GTSRB CompactCNN have?")["answer"]
    assert "y_train_onehot" in P.ask("Which line was added to fix the NumPy notebook crash?")["answer"]
    assert "verified identical to the official" in P.ask("Was the GTSRB test split verified identical to the official one?")["answer"]
    assert "NOT treated as permission" in P.ask("Is the Kaggle CC0 tag treated as permission to redistribute BBC articles?")["answer"]

def test_dev_out_of_scope_abstains():
    for q in ["What is the training time of the BBC search index on a GPU?",
              "What is the accuracy of the loan dashboard's approval prediction model?",
              "Which transformer model powers the image caption search?",
              "how do I write a SQL join",
              "how many hotels are there in Cairo"]:
        assert P.ask(q)["abstained"], q


def test_n23_bbc_sole_authorship_states_unconfirmed_from_bbc_doc():
    r = P.ask("Is Ahmed confirmed as the sole author of the BBC search coursework?")
    assert not r["abstained"]
    assert all(c.startswith("bbc-article-search#") for c in r["citations"])
    assert re.search(r"unconfirmed|not confirmed|contribution", r["answer"], re.I)
    assert not r["answer"].strip().lower().startswith("yes")


def test_authorship_status_routes_to_each_projects_own_doc():
    for q, doc in [
        ("Is Ahmed the sole author of the image caption search project?", "image-caption-search"),
        ("What is Ahmed's personal contribution to the CIFAR-10 project?", "cifar10-deep-learning"),
        ("Did Ahmed write the traffic sign classifier alone?", "traffic-sign-gtsrb"),
    ]:
        r = P.ask(q)
        assert not r["abstained"], q
        assert all(c.startswith(doc + "#") for c in r["citations"]), (q, r["citations"])
        assert "unconfirmed" in r["answer"].lower(), q


def test_metric_questions_do_not_return_contribution_status_text():
    for q in ["what accuracy did the traffic sign model reach?",
              "What AUC did the food price spike model reach?",
              "How many TXT files did the BBC download include?"]:
        r = P.ask(q)
        assert "contribution status" not in r["answer"].lower(), q
        assert "sole authorship" not in r["answer"].lower(), q


def test_contribution_status_is_metadata_not_corpus_text():
    from retriever import Retriever as R
    assert all("Contribution status:" not in c["text"] for c in R().chunks)


def test_ahmed_mentioning_ooc_question_behaves_like_r2_baseline():
    r = P.ask("what is Ahmed's favorite food")
    assert r["abstained"]


def test_authorship_status_answered_for_doc_without_verbatim_sentence():
    r = P.ask("What is Ahmed's personal contribution to the CIFAR-10 project?")
    assert not r["abstained"]
    assert "unconfirmed" in r["answer"].lower()
    assert all(c.startswith("cifar10-deep-learning#") for c in r["citations"])


# ---- comparison-question feature (frozen criteria: eval/v25/COMPARISON_CRITERIA.md) ----
import re as _re

ROOT = Path(__file__).resolve().parent.parent

def test_compare_two_projects_labels_both_sides():
    P = globals()['P']
    a = P.ask("Compare the accuracy of the traffic sign project and the CIFAR deep learning project")
    assert not a["abstained"] and a["why"] == "comparison"
    assert "traffic-sign-gtsrb:" in a["answer"] and "cifar10-deep-learning:" in a["answer"]
    docs_cited = {c.split('#')[0] for c in a["citations"]}
    assert "traffic-sign-gtsrb" in docs_cited and "cifar10-deep-learning" in docs_cited

def test_compare_sentences_verbatim_per_doc():
    P = globals()['P']
    a = P.ask("compare gtsrb and food price spike model accuracy")
    assert not a["abstained"]
    docs = ["traffic-sign-gtsrb", "egypt-food-prices-ml"]
    for i, doc in enumerate(docs):
        start = a["answer"].index(doc + ": ") + len(doc) + 2
        end = a["answer"].index(docs[i + 1] + ": ") if i + 1 < len(docs) else len(a["answer"])
        seg = a["answer"][start:end]
        body = open(str(ROOT / 'corpus' / (doc + '.md'))).read()
        from extractive import sentences as _sents
        for sent in _sents(seg):
            assert sent.rstrip('.') in body, (doc, sent)

def test_compare_authorship_excluded():
    P = globals()['P']
    a = P.ask("who built the traffic sign and the CIFAR projects, and was it alone")
    assert a["why"] != "comparison"  # AUTHOR_Q keeps the single-doc authorship path

def test_compare_metric_absent_from_both_abstains():
    P = globals()['P']
    a = P.ask("Compare the salary of the traffic sign project and the CIFAR project")
    assert a["abstained"]

def test_compare_no_synthesis_connective():
    P = globals()['P']
    a = P.ask("Compare the accuracy of the traffic sign project and the CIFAR deep learning project")
    # no comparative judgment introduced between the two labeled parts
    assert not _re.search(r"\bbetter than|\bworse than|\boutperform", a["answer"], _re.I)


def test_comparison_skips_historical_marker_sentences():
    # r8: comparison shows current status only; a sentence carrying an explicit
    # HISTORICAL marker inside a non-historical section must not be quoted as a
    # current result (regression: GTSRB duplicate-audit sentence).
    a = P.ask("Compare the accuracy of the traffic sign project and the CIFAR deep learning project")
    assert a["why"] == "comparison"
    assert "98.8915" in a["answer"] and "89.51" in a["answer"]
    assert "accuracy excluding it" not in a["answer"]


def test_historical_question_still_surfaces_audit_sentence():
    # the historical sentence stays reachable for historical questions, labeled
    a = P.ask("What was the accuracy excluding the duplicate in the historical audit?")
    assert "accuracy excluding it" in a["answer"]
    assert "(historical)" in a["answer"]
