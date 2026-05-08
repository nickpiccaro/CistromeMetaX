"""
CistromeMetaX Test Suite

Tests are organized into tiers:
  1. Import & dependency tests — verify all packages resolve correctly
  2. Utility function tests — pure logic, no LLM or network calls
  3. XML parsing tests — test GSM/GSE XML simplification
  4. Validation tests — test factor & ontology validation against local data
  5. CLI tests — test argument parsing and entry point wiring
  6. Integration tests (LLM) — require an API key, skipped when unavailable

Run all non-LLM tests:
    python -m pytest tests/test_cistromemetax.py -v

Run including LLM integration tests (needs OPENAI_API_KEY in .env or env):
    python -m pytest tests/test_cistromemetax.py -v --run-llm
"""

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "tests" / "test_data"
SAMPLE_GSM_XML = TEST_DATA_DIR / "sample_gsm.xml"
SAMPLE_GSE_XML = TEST_DATA_DIR / "sample_gse.xml"
DATA_DIR = REPO_ROOT / "CistromeMetaX" / "data"
PARSED_FACTOR_DIR = DATA_DIR / "parsed_factor_data"
PARSED_ONTOLOGY_DIR = DATA_DIR / "parsed_ontology_data"


# ===================================================================
# 1. IMPORT & DEPENDENCY TESTS
# ===================================================================
class TestImports:
    """Verify that all package imports resolve without errors."""

    def test_import_langchain_core_messages(self):
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        assert HumanMessage is not None
        assert SystemMessage is not None
        assert AIMessage is not None

    def test_import_langchain_core_prompts(self):
        from langchain_core.prompts import ChatPromptTemplate
        assert ChatPromptTemplate is not None

    def test_import_langchain_init_chat_model(self):
        from langchain.chat_models import init_chat_model
        assert init_chat_model is not None

    def test_import_langchain_openai(self):
        import langchain_openai
        assert langchain_openai is not None

    def test_import_langchain_anthropic(self):
        import langchain_anthropic
        assert langchain_anthropic is not None

    def test_import_langchain_google_genai(self):
        import langchain_google_genai
        assert langchain_google_genai is not None

    def test_import_langchain_mistralai(self):
        import langchain_mistralai
        assert langchain_mistralai is not None

    def test_import_langchain_community(self):
        import langchain_community
        assert langchain_community is not None

    def test_import_langchain_experimental(self):
        import langchain_experimental
        assert langchain_experimental is not None

    def test_import_pydantic(self):
        from pydantic import BaseModel, Field
        assert BaseModel is not None

    def test_import_rapidfuzz(self):
        from rapidfuzz import fuzz
        assert fuzz is not None

    def test_import_rdflib(self):
        import rdflib
        assert rdflib is not None

    def test_import_requests(self):
        import requests
        assert requests is not None

    def test_import_dotenv(self):
        from dotenv import load_dotenv
        assert load_dotenv is not None

    def test_import_cistromemetax_package(self):
        """Top-level package import works."""
        from CistromeMetaX import (
            meta_extract_factors,
            meta_extract_ontologies,
            meta_extract_factors_and_ontologies,
        )
        assert callable(meta_extract_factors)
        assert callable(meta_extract_ontologies)
        assert callable(meta_extract_factors_and_ontologies)

    def test_import_parser_extractor_submodule(self):
        from CistromeMetaX.parser_extractor import (
            clean_input,
            clean_input_fuzzy,
            remove_words,
            process_string,
            collapse_ontology_terms,
            simplify_gsm_xml_file,
            simplify_gse_xml_file,
            validate_histone_mark,
            match_human_gene,
            validate_transcription_factor,
            validate_chromatin_remodelers,
            _parse_llm_json,
            _parse_llm_list,
            _format_output_structure,
        )
        # Smoke check that they're all callable
        for fn in [
            clean_input, clean_input_fuzzy, remove_words, process_string,
            collapse_ontology_terms, simplify_gsm_xml_file, simplify_gse_xml_file,
            validate_histone_mark, match_human_gene, validate_transcription_factor,
            validate_chromatin_remodelers, _parse_llm_json, _parse_llm_list,
            _format_output_structure,
        ]:
            assert callable(fn)


# ===================================================================
# 2. UTILITY FUNCTION TESTS
# ===================================================================
class TestCleanInput:
    def test_basic(self):
        from CistromeMetaX.parser_extractor import clean_input
        assert clean_input("Hello World!") == "helloworld"

    def test_none(self):
        from CistromeMetaX.parser_extractor import clean_input
        assert clean_input(None) == ""

    def test_special_chars(self):
        from CistromeMetaX.parser_extractor import clean_input
        assert clean_input("H3K27ac") == "h3k27ac"

    def test_whitespace_newlines(self):
        from CistromeMetaX.parser_extractor import clean_input
        assert clean_input("  foo\nbar  ") == "foobar"


class TestCleanInputFuzzy:
    def test_preserves_spaces(self):
        from CistromeMetaX.parser_extractor import clean_input_fuzzy
        result = clean_input_fuzzy("Hello, World!")
        assert result == "hello world"

    def test_none(self):
        from CistromeMetaX.parser_extractor import clean_input_fuzzy
        assert clean_input_fuzzy(None) == ""


class TestRemoveWords:
    def test_removes_stop_words(self):
        from CistromeMetaX.parser_extractor import remove_words as rw
        result = rw("primary cell line of human tissue")
        # "primary", "cell", "line", "of", "human", "tissue" should all be removed
        assert "primary" not in result
        assert "cell" not in result
        assert "human" not in result

    def test_preserves_important_words(self):
        from CistromeMetaX.parser_extractor import remove_words as rw
        result = rw("fibroblast cell")
        assert "fibroblast" in result


class TestProcessString:
    def test_basic(self):
        from CistromeMetaX.parser_extractor import process_string
        result = process_string("Hello World")
        assert result == "helloworld"

    def test_with_remove(self):
        from CistromeMetaX.parser_extractor import process_string
        result = process_string("primary cell line", remove=True)
        # all three words are stop words, so result should be empty or None-ish
        assert result is not None

    def test_none_input(self):
        from CistromeMetaX.parser_extractor import process_string
        assert process_string(None) is None

    def test_empty_string(self):
        from CistromeMetaX.parser_extractor import process_string
        assert process_string("") is None


class TestCollapseOntologyTerms:
    def test_single_entry(self):
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        entries = [
            {"official_term": "fibroblast", "ontology_type": "EFO", "ontology_accession": "EFO_001"}
        ]
        result = collapse_ontology_terms(entries)
        assert len(result) == 1
        assert result[0]["official_term"] == "fibroblast"

    def test_merge_same_term_different_ontology_type(self):
        """Same official_term and accession from different ontology sources should merge."""
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        entries = [
            {"official_term": "lung", "ontology_type": "EFO", "ontology_accession": "UBERON_0002048"},
            {"official_term": "lung", "ontology_type": "Uberon", "ontology_accession": "UBERON_0002048"},
        ]
        result = collapse_ontology_terms(entries)
        assert len(result) == 1
        assert result[0]["official_term"] == "lung"
        ot = result[0]["ontology_type"]
        assert isinstance(ot, list)
        assert set(ot) == {"EFO", "Uberon"}

    def test_same_term_different_accession_not_collapsed(self):
        """Same official_term but different accessions are distinct entries."""
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        entries = [
            {"official_term": "lung", "ontology_type": "EFO", "ontology_accession": "EFO_100"},
            {"official_term": "lung", "ontology_type": "Uberon", "ontology_accession": "UBERON_200"},
        ]
        result = collapse_ontology_terms(entries)
        assert len(result) == 2

    def test_distinct_terms_not_collapsed(self):
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        entries = [
            {"official_term": "lung", "ontology_type": "EFO", "ontology_accession": "EFO_100"},
            {"official_term": "brain", "ontology_type": "Uberon", "ontology_accession": "UBERON_001"},
        ]
        result = collapse_ontology_terms(entries)
        assert len(result) == 2

    def test_fully_identical_entries_deduplicated(self):
        """Exact duplicate entries (same accession appearing multiple times) should
        be reduced to one — this is the core reviewer-reported bug."""
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        entry = {
            "official_term": "HeLa",
            "ontology_accession": "CVCL_0030",
            "ontology_type": "Cellosaurus",
            "term": "HeLa",
            "term_identity": "cell_line",
        }
        # Simulate the bug: same entry appearing 3 times
        entries = [entry.copy(), entry.copy(), entry.copy()]
        result = collapse_ontology_terms(entries)
        assert len(result) == 1
        assert result[0]["ontology_accession"] == "CVCL_0030"

    def test_duplicate_with_list_ontology_type(self):
        """Entries where ontology_type is already a list should not crash or duplicate."""
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        entries = [
            {"official_term": "fibroblast", "ontology_accession": "CL_0000057",
             "ontology_type": ["EFO", "Uberon"], "term_identity": "cell_type"},
            {"official_term": "fibroblast", "ontology_accession": "CL_0000057",
             "ontology_type": ["EFO", "Uberon"], "term_identity": "cell_type"},
        ]
        result = collapse_ontology_terms(entries)
        assert len(result) == 1

    def test_realistic_cellosaurus_triplicates(self):
        """Simulate the exact reviewer scenario: same CVCL entry from strict,
        reduced, and fuzzy index lookups producing 3 identical matches."""
        from CistromeMetaX.parser_extractor import collapse_ontology_terms
        base = {
            "official_term": "T-47D",
            "ontology_accession": "CVCL_2096",
            "ontology_type": "Cellosaurus",
            "term": "T47D",
            "term_identity": "cell_line",
        }
        entries = [base.copy(), base.copy(), base.copy()]
        result = collapse_ontology_terms(entries)
        assert len(result) == 1
        assert result[0]["ontology_accession"] == "CVCL_2096"


class TestFormatOutputStructure:
    def test_factor_only(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        result = _format_output_structure("GSM123", extracted_factor="H3K27ac")
        assert result == {"GSM123": {"factor": {"extracted_factor": "H3K27ac"}}}
        # No factor_status when factor is successfully extracted
        assert "factor_status" not in result["GSM123"]["factor"]

    def test_ontology_only(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        ont = {"cell_line": [{"official_term": "MCF7"}], "cell_type": None, "tissue": None, "disease": "N/A"}
        result = _format_output_structure("GSM456", extracted_ontology=ont)
        assert "ontology" in result["GSM456"]
        assert result["GSM456"]["ontology"]["extracted_ontologies"]["disease"] == "N/A"

    def test_both(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        ont = {"cell_line": None, "cell_type": None, "tissue": None, "disease": None}
        result = _format_output_structure("GSM789", extracted_factor="ER", extracted_ontology=ont)
        assert "factor" in result["GSM789"]
        assert "ontology" in result["GSM789"]

    def test_control_sample_status(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        result = _format_output_structure("GSM100", extracted_factor="N/A", factor_status="control_sample")
        factor = result["GSM100"]["factor"]
        assert factor["extracted_factor"] == "N/A"
        assert factor["factor_status"] == "control_sample"

    def test_verification_failed_status(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        result = _format_output_structure("GSM200", extracted_factor="N/A", factor_status="verification_failed")
        factor = result["GSM200"]["factor"]
        assert factor["extracted_factor"] == "N/A"
        assert factor["factor_status"] == "verification_failed"

    def test_no_status_when_factor_present(self):
        """factor_status should not appear when extracted_factor is not N/A."""
        from CistromeMetaX.parser_extractor import _format_output_structure
        result = _format_output_structure("GSM300", extracted_factor="TP53", factor_status="control_sample")
        factor = result["GSM300"]["factor"]
        assert factor["extracted_factor"] == "TP53"
        assert "factor_status" not in factor

    def test_extraction_failed_status(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        result = _format_output_structure("GSM400", extracted_factor="N/A", factor_status="extraction_failed")
        factor = result["GSM400"]["factor"]
        assert factor["factor_status"] == "extraction_failed"

    def test_no_factor_detected_status(self):
        from CistromeMetaX.parser_extractor import _format_output_structure
        result = _format_output_structure("GSM500", extracted_factor="N/A", factor_status="no_factor_detected")
        factor = result["GSM500"]["factor"]
        assert factor["factor_status"] == "no_factor_detected"


# ===================================================================
# 3. LLM RESPONSE PARSING TESTS
# ===================================================================
class TestParseLlmJson:
    def test_clean_json(self):
        from CistromeMetaX.parser_extractor import _parse_llm_json
        result = _parse_llm_json('{"factor": "ER", "reasoning": "test"}')
        assert result["factor"] == "ER"

    def test_markdown_fenced(self):
        from CistromeMetaX.parser_extractor import _parse_llm_json
        content = '```json\n{"factor": "FOXA1"}\n```'
        result = _parse_llm_json(content)
        assert result["factor"] == "FOXA1"

    def test_text_preamble(self):
        from CistromeMetaX.parser_extractor import _parse_llm_json
        content = 'Here is the result:\n{"factor": "H3K27ac", "reasoning": "..."}'
        result = _parse_llm_json(content)
        assert result["factor"] == "H3K27ac"

    def test_empty_raises(self):
        from CistromeMetaX.parser_extractor import _parse_llm_json
        with pytest.raises(ValueError):
            _parse_llm_json("")

    def test_no_json_raises(self):
        from CistromeMetaX.parser_extractor import _parse_llm_json
        with pytest.raises(ValueError):
            _parse_llm_json("no json here at all")


class TestParseLlmList:
    def test_clean_list(self):
        from CistromeMetaX.parser_extractor import _parse_llm_list
        result = _parse_llm_list('["ESR1", "ER"]')
        assert result == ["ESR1", "ER"]

    def test_markdown_fenced(self):
        from CistromeMetaX.parser_extractor import _parse_llm_list
        content = '```json\n["FOXA1", "HNF3A"]\n```'
        result = _parse_llm_list(content)
        assert "FOXA1" in result

    def test_python_style_single_quotes(self):
        from CistromeMetaX.parser_extractor import _parse_llm_list
        content = "['ESR1', 'ER']"
        result = _parse_llm_list(content)
        assert "ESR1" in result

    def test_empty_raises(self):
        from CistromeMetaX.parser_extractor import _parse_llm_list
        with pytest.raises(ValueError):
            _parse_llm_list("")


# ===================================================================
# 4. XML PARSING TESTS
# ===================================================================
class TestSimplifyGsmXml:
    def test_simplify_sample_gsm(self):
        from CistromeMetaX.parser_extractor import simplify_gsm_xml_file
        result = simplify_gsm_xml_file(str(SAMPLE_GSM_XML))
        assert "GSM534473" in result
        assert "H3K4me2" in result
        assert "Homo sapiens" in result
        # Excluded elements should not appear
        assert "Platform" not in result
        assert "Supplementary-Data" not in result

    def test_contains_channel_data(self):
        from CistromeMetaX.parser_extractor import simplify_gsm_xml_file
        result = simplify_gsm_xml_file(str(SAMPLE_GSM_XML))
        assert "Channel" in result
        assert "hASC" in result


class TestSimplifyGseXml:
    def test_simplify_sample_gse(self):
        from CistromeMetaX.parser_extractor import simplify_gse_xml_file
        result = simplify_gse_xml_file(str(SAMPLE_GSE_XML))
        assert "GSE20752" in result
        assert "adipose" in result.lower() or "chromatin" in result.lower()
        # Excluded elements should not appear
        assert "Contributor-Ref" not in result
        assert "Sample-Ref" not in result


class TestRemoveInvalidCharacters:
    def test_valid_xml(self):
        from CistromeMetaX.parser_extractor import remove_invalid_characters_from_file
        result = remove_invalid_characters_from_file(str(SAMPLE_GSM_XML))
        assert result is not None
        assert "GSM534473" in result

    def test_nonexistent_file(self):
        from CistromeMetaX.parser_extractor import remove_invalid_characters_from_file
        result = remove_invalid_characters_from_file("/nonexistent/path.xml")
        assert result is None


# ===================================================================
# 5. HISTONE MARK VALIDATION TESTS
# ===================================================================
class TestValidateHistoneMark:
    def test_valid_marks(self):
        from CistromeMetaX.parser_extractor import validate_histone_mark
        valid_marks = [
            "H3K27ac", "H3K4me1", "H3K4me2", "H3K4me3",
            "H3K27me3", "H3K36me3", "H3K9me3", "H3K9ac",
            "H4K20me1", "H2AK119ub",
        ]
        for mark in valid_marks:
            assert validate_histone_mark(mark), f"{mark} should be valid"

    def test_invalid_marks(self):
        from CistromeMetaX.parser_extractor import validate_histone_mark
        invalid_marks = ["ER", "FOXA1", "MCF7", "K27ac", "foobar"]
        for mark in invalid_marks:
            assert not validate_histone_mark(mark), f"{mark} should be invalid"

    def test_multi_ptm(self):
        from CistromeMetaX.parser_extractor import validate_histone_mark
        assert validate_histone_mark("H3K27acK36me3")


# ===================================================================
# 6. FACTOR VALIDATION TESTS (against local data files)
# ===================================================================
@pytest.fixture(scope="module")
def factor_data():
    """Load the parsed factor validation data files."""
    if not PARSED_FACTOR_DIR.exists():
        pytest.skip("Parsed factor data not available (run cistromeMX-update_data first)")
    genes_df = pd.read_csv(PARSED_FACTOR_DIR / "gene_info.csv", encoding="utf-8")
    tf_df = pd.read_csv(PARSED_FACTOR_DIR / "Homo_sapiens_TF.csv", sep="\t", encoding="utf-8")
    cr_df = pd.read_csv(PARSED_FACTOR_DIR / "Homo_sapiens_CR.csv", encoding="utf-8")
    return genes_df, tf_df, cr_df


class TestMatchHumanGene:
    def test_find_known_gene(self, factor_data):
        from CistromeMetaX.parser_extractor import match_human_gene
        genes_df, _, _ = factor_data
        matches = match_human_gene(genes_df, "TP53")
        assert len(matches) >= 1
        symbols = [m["Symbol"] for m in matches]
        assert "TP53" in symbols

    def test_no_match(self, factor_data):
        from CistromeMetaX.parser_extractor import match_human_gene
        genes_df, _, _ = factor_data
        matches = match_human_gene(genes_df, "ZZZZNOTAREALGENE")
        assert len(matches) == 0

    def test_missing_columns_raises(self):
        from CistromeMetaX.parser_extractor import match_human_gene
        bad_df = pd.DataFrame({"col1": [1], "col2": [2]})
        with pytest.raises(ValueError):
            match_human_gene(bad_df, "TP53")


class TestValidateTranscriptionFactor:
    def test_valid_tf(self, factor_data):
        from CistromeMetaX.parser_extractor import validate_transcription_factor
        _, tf_df, _ = factor_data
        # Get a known TF symbol from the database
        known_tf = tf_df["Symbol"].iloc[0]
        matching = [{"Symbol": known_tf}]
        result = validate_transcription_factor(matching, tf_df)
        assert len(result) == 1

    def test_invalid_tf(self, factor_data):
        from CistromeMetaX.parser_extractor import validate_transcription_factor
        _, tf_df, _ = factor_data
        matching = [{"Symbol": "NOTAREALTF999"}]
        result = validate_transcription_factor(matching, tf_df)
        assert len(result) == 0


class TestValidateChromatinRemodelers:
    def test_known_remodeler(self, factor_data):
        from CistromeMetaX.parser_extractor import validate_chromatin_remodelers
        _, _, cr_df = factor_data
        # Build a dict mimicking the chromatin DB structure
        cr_db = {
            "chromatin remodeling": cr_df["chromatin_remodeler"].tolist(),
            "synonyms": cr_df["synonyms"].tolist(),
        }
        known_cr = cr_df["chromatin_remodeler"].iloc[0]
        matching = [{"Symbol": known_cr}]
        result = validate_chromatin_remodelers(matching, cr_db)
        assert len(result) >= 1


# ===================================================================
# 7. ONTOLOGY VALIDATION TESTS (against local data files)
# ===================================================================
@pytest.fixture(scope="module")
def ontology_indices():
    """Load the parsed ontology index files."""
    if not PARSED_ONTOLOGY_DIR.exists():
        pytest.skip("Parsed ontology data not available (run cistromeMX-update_data first)")
    with open(PARSED_ONTOLOGY_DIR / "cellosaurus.json") as f:
        cellosaurus = json.load(f)
    with open(PARSED_ONTOLOGY_DIR / "efo.json") as f:
        efo = json.load(f)
    with open(PARSED_ONTOLOGY_DIR / "uberon.json") as f:
        uberon = json.load(f)
    return cellosaurus, efo, uberon


class TestValidateOntology:
    def test_known_cell_line(self, ontology_indices):
        from CistromeMetaX.parser_extractor import validate_ontology
        cellosaurus, efo, uberon = ontology_indices
        extracted = {"cell_line": "MCF-7", "cell_type": "N/A", "tissue": "N/A", "disease": "N/A"}
        result = validate_ontology(extracted, cellosaurus, efo, uberon)
        # MCF-7 should match in cellosaurus
        assert result["cell_line"] is not None
        assert len(result["cell_line"]) >= 1

    def test_na_fields_stay_none(self, ontology_indices):
        from CistromeMetaX.parser_extractor import validate_ontology
        cellosaurus, efo, uberon = ontology_indices
        extracted = {"cell_line": "N/A", "cell_type": "N/A", "tissue": "N/A", "disease": "N/A"}
        result = validate_ontology(extracted, cellosaurus, efo, uberon)
        assert result["cell_line"] is None
        assert result["cell_type"] is None
        assert result["tissue"] is None
        assert result["disease"] is None


# ===================================================================
# 8. CLI TESTS
# ===================================================================
class TestCLI:
    def test_meta_extract_entry_point_exists(self):
        """The cistromeMX-extract entry point should resolve to cli.meta_extract."""
        from CistromeMetaX.cli import meta_extract
        assert callable(meta_extract)

    def test_update_data_entry_point_exists(self):
        """The cistromeMX-update_data entry point should resolve to cli.update_data."""
        from CistromeMetaX.cli import update_data
        assert callable(update_data)

    def test_parse_gsm_ids_from_file(self):
        from CistromeMetaX.cli import _parse_gsm_ids_input
        result = _parse_gsm_ids_input(str(TEST_DATA_DIR / "gsm_ids.json"))
        # CLI parser returns the file path string for valid files (loaded later)
        assert result == str(TEST_DATA_DIR / "gsm_ids.json")

    def test_parse_gsm_ids_from_json_string(self):
        from CistromeMetaX.cli import _parse_gsm_ids_input
        result = _parse_gsm_ids_input('["GSM123456", "GSM789012"]')
        assert result == ["GSM123456", "GSM789012"]

    def test_parse_gsm_ids_from_list(self):
        from CistromeMetaX.cli import _parse_gsm_ids_input
        result = _parse_gsm_ids_input(["GSM111", "GSM222"])
        assert result == ["GSM111", "GSM222"]

    def test_cli_help_text_uses_correct_command_name(self):
        """Verify the help text references cistromeMX-extract, not geoMX-extract."""
        from CistromeMetaX import cli
        import inspect
        source = inspect.getsource(cli)
        assert "geoMX-extract" not in source
        assert "geoMX-update_data" not in source


# ===================================================================
# 9. JSON DATA LOADING TESTS
# ===================================================================
class TestLoadJsonData:
    def test_load_valid_json(self):
        from CistromeMetaX.parser_extractor import _load_json_data
        result = _load_json_data(str(TEST_DATA_DIR / "gsm_ids.json"))
        assert result == ["GSM534473"]

    def test_load_nonexistent_file(self):
        from CistromeMetaX.parser_extractor import _load_json_data
        result = _load_json_data("/nonexistent/path.json")
        assert result is None

    def test_load_mapping_file(self):
        from CistromeMetaX.parser_extractor import _load_json_data
        result = _load_json_data(str(TEST_DATA_DIR / "gsm_to_gse.json"))
        assert isinstance(result, dict)
        assert "GSM534473" in result
        assert result["GSM534473"] == ["GSE20752"]


class TestParseGsmIdsInput:
    def test_file_path(self):
        from CistromeMetaX.parser_extractor import _parse_gsm_ids_input
        result = _parse_gsm_ids_input(str(TEST_DATA_DIR / "gsm_ids.json"))
        assert result == ["GSM534473"]

    def test_list_input(self):
        from CistromeMetaX.parser_extractor import _parse_gsm_ids_input
        result = _parse_gsm_ids_input(["GSM111", "GSM222"])
        assert result == ["GSM111", "GSM222"]

    def test_pandas_series(self):
        from CistromeMetaX.parser_extractor import _parse_gsm_ids_input
        series = pd.Series(["GSM100", "GSM200"])
        result = _parse_gsm_ids_input(series)
        assert result == ["GSM100", "GSM200"]


# ===================================================================
# 10. INTEGRATION TESTS (require LLM API key)
# ===================================================================
@pytest.mark.llm
class TestLLMIntegration:
    """Tests that make actual LLM calls. Skipped by default."""

    def test_init_llm_default(self):
        from CistromeMetaX.parser_extractor import _init_llm
        llm = _init_llm()
        assert llm is not None

    def test_is_control(self):
        from CistromeMetaX.parser_extractor import is_control, simplify_gsm_xml_file
        gsm_str = simplify_gsm_xml_file(str(SAMPLE_GSM_XML))
        result = is_control(gsm_str)
        assert result in ("True", "False")
        # The sample is H3K4me2 ChIP — not a control
        assert result == "False"

    def test_extract_factor(self):
        from CistromeMetaX.parser_extractor import (
            extract_factor, simplify_gsm_xml_file, simplify_gse_xml_file
        )
        gsm_str = simplify_gsm_xml_file(str(SAMPLE_GSM_XML))
        gse_str = simplify_gse_xml_file(str(SAMPLE_GSE_XML))
        result = extract_factor(gsm_str, [gse_str])
        assert result is not None
        # The sample is targeting H3K4me2
        assert "H3K4" in result or "me2" in result.lower()

    def test_extract_structured_ontology(self):
        from CistromeMetaX.parser_extractor import (
            extract_structured_ontology, simplify_gsm_xml_file, simplify_gse_xml_file
        )
        gsm_str = simplify_gsm_xml_file(str(SAMPLE_GSM_XML))
        gse_str = simplify_gse_xml_file(str(SAMPLE_GSE_XML))
        result = extract_structured_ontology(gsm_str, [gse_str])
        assert hasattr(result, "cell_line") or isinstance(result, dict)

    def test_generate_synonyms(self):
        from CistromeMetaX.parser_extractor import generate_synonyms
        result = generate_synonyms("TP53")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_meta_extract_factors_end_to_end(self):
        """Full pipeline test using test fixtures."""
        from CistromeMetaX import meta_extract_factors
        result = meta_extract_factors(
            gsm_ids_input=str(TEST_DATA_DIR / "gsm_ids.json"),
            gsm_to_gse_path=str(TEST_DATA_DIR / "gsm_to_gse.json"),
            gsm_paths_path=str(TEST_DATA_DIR / "gsm_paths.json"),
            gse_paths_path=str(TEST_DATA_DIR / "gse_paths.json"),
        )
        assert isinstance(result, (list, dict))
