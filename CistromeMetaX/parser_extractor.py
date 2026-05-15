import re
import pandas as pd
import sys
import xml.etree.ElementTree as ET
import json
import ast
import os
from pathlib import Path
from rapidfuzz import fuzz
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from .geo_fetch import (
    fetch_geo_xml,
    discover_parent_gses,
    expand_gse_to_gsms,
    MAX_TRIES as _GEO_MAX_TRIES,
)

# Default model used when no model is specified
_DEFAULT_MODEL = "openai:gpt-4o-mini"


def _init_llm(model=None):
    """
    Initialize a chat model using langchain's init_chat_model.
    
    Args:
        model (str, optional): Model identifier in the format "provider:model_name"
            (e.g., "openai:gpt-4o-mini", "anthropic:claude-sonnet-4-5-20250929").
            If only a model name is provided, the provider will be inferred.
            Defaults to "openai:gpt-4o-mini" if None.
    
    Returns:
        BaseChatModel: An initialized chat model instance.
    
    Raises:
        ValueError: If the model or provider is invalid.
        RuntimeError: If the required API key is missing or authentication fails.
    """
    load_dotenv(dotenv_path=Path.cwd() / ".env")

    model = model or _DEFAULT_MODEL
    
    try:
        llm = init_chat_model(model, temperature=0)
        return llm
    except ImportError as e:
        # Missing provider package (e.g., langchain-anthropic not installed)
        provider = model.split(":")[0] if ":" in model else "unknown"
        raise ImportError(
            f"The integration package for provider '{provider}' is not installed. "
            f"Please install it (e.g., 'pip install langchain-{provider}'). "
            f"Original error: {e}"
        ) from e
    except Exception as e:
        error_msg = str(e).lower()
        if "api_key" in error_msg or "api key" in error_msg or "authenticate" in error_msg or "auth" in error_msg:
            provider = model.split(":")[0] if ":" in model else "unknown"
            raise RuntimeError(
                f"Authentication failed for model '{model}'. "
                f"Please ensure the correct API key for '{provider}' is set in your .env file. "
                f"Common environment variable names: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                f"GOOGLE_API_KEY, etc. Original error: {e}"
            ) from e
        elif "not found" in error_msg or "invalid" in error_msg or "does not exist" in error_msg:
            raise ValueError(
                f"Invalid model '{model}'. Please check the model name and provider. "
                f"Use the format 'provider:model_name' (e.g., 'openai:gpt-4o-mini', "
                f"'anthropic:claude-sonnet-4-5-20250929'). Original error: {e}"
            ) from e
        else:
            raise


def _parse_llm_json(content):
    """
    Robustly parse a JSON object from an LLM response string.
    
    Handles common LLM output variations:
      - Clean JSON: {"factor": "ER", "reasoning": "..."}
      - Markdown-wrapped: ```json\n{"factor": "ER"}\n```
      - Text preamble: "Here is the result:\n{"factor": "ER"}"
    
    Args:
        content (str): Raw LLM response content.
    
    Returns:
        dict: Parsed JSON object.
    
    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    if not content or not content.strip():
        raise ValueError("Empty LLM response")
    
    text = content.strip()
    
    # 1. Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    import re
    fenced = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 3. Find the first { ... } block in the text
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def _parse_llm_list(content):
    """
    Robustly parse a JSON list/array from an LLM response string.
    
    Handles common LLM output variations:
      - Clean list: ["synonym1", "synonym2"]
      - Markdown-wrapped: ```json\n["synonym1"]\n```
      - Text preamble: "Here are the synonyms:\n["synonym1", "synonym2"]"
      - Python-style lists with single quotes: ['synonym1', 'synonym2']
    
    Args:
        content (str): Raw LLM response content.
    
    Returns:
        list: Parsed list.
    
    Raises:
        ValueError: If no valid list can be extracted.
    """
    if not content or not content.strip():
        raise ValueError("Empty LLM response")
    
    text = content.strip()
    
    # 1. Try direct JSON parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # 2. Strip markdown code fences
    import re
    fenced = re.search(r'```(?:json|python)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # Try ast.literal_eval for Python-style lists
        try:
            result = ast.literal_eval(inner)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    
    # 3. Find the first [ ... ] block in the text
    bracket_match = re.search(r'\[.*\]', text, re.DOTALL)
    if bracket_match:
        candidate = bracket_match.group(0)
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
        # Try ast.literal_eval for single-quoted Python lists
        try:
            result = ast.literal_eval(candidate)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass
    
    raise ValueError(f"Could not parse list from LLM response: {text[:200]}")


def _make_cached_system_message(text, model):
    """
    Build a SystemMessage that opts into prompt caching where the provider
    supports it explicitly (Anthropic). For every other LangChain-supported
    provider (OpenAI, Google GenAI, DeepSeek, Mistral, ...) prompt caching is
    either applied automatically server-side or simply unavailable, so a plain
    text SystemMessage is the correct and safe representation.

    Why provider branching: ``langchain-anthropic``'s ``ChatAnthropic`` honors
    ``cache_control`` markers attached to text content blocks and forwards them
    to Anthropic, which yields ~90% input-token discounts on cache hits. Other
    LangChain integrations may not know what to do with a ``cache_control`` key
    inside a content-block list, so we hand them a plain string.

    Static prefix discipline matters across ALL providers: OpenAI auto-caches
    any prefix >=1024 tokens, Google Gemini caches implicitly, and DeepSeek
    has automatic context caching. Keeping the guidelines block byte-identical
    across calls is what unlocks those benefits.

    Args:
        text (str): Static system-prompt text. Must be byte-identical across
            calls that should share the same cache entry.
        model (str): The same ``provider:model`` string passed to ``_init_llm``.

    Returns:
        SystemMessage: Ready to send to ``llm.invoke([...])``.
    """
    provider = (model or _DEFAULT_MODEL).split(":", 1)[0].lower()
    if provider == "anthropic":
        return SystemMessage(content=[{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }])
    return SystemMessage(content=text)


def _log_cache_usage(response, label=""):
    """
    Print LLM response ``usage_metadata`` to stderr when the
    ``CISTROMEMX_CACHE_DEBUG`` environment variable is set. Lets users confirm
    that prompt caching is firing for their provider without changing any
    function signatures. No-op when the env var is unset.

    Anthropic populates ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens``; OpenAI surfaces cached tokens under
    ``input_token_details``. Both show up in ``usage_metadata`` on the
    LangChain response object.
    """
    if not os.environ.get("CISTROMEMX_CACHE_DEBUG"):
        return
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        usage = getattr(response, "response_metadata", {}).get("usage") or getattr(response, "response_metadata", {}).get("token_usage")
    if usage:
        prefix = f"[cache:{label}] " if label else "[cache] "
        print(f"{prefix}{usage}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Module-level prompt constants used by extract_factor, factor_recheck, and
# extract_structured_ontology.
#
# These were previously inlined inside each function. Hoisting them to module
# scope makes the strings byte-identical across calls (so OpenAI/Gemini/
# DeepSeek auto-caching has something to hash) and lets extract_factor and
# factor_recheck SHARE the same cached prefix on Anthropic. The dynamic
# "previously failed factors" context that used to live inside the factor
# recheck system prompt now lives in the human input template, so the system
# prompt is identical for both call sites.
#
# These strings are inlined into SystemMessage / HumanMessage directly and so
# use single braces (no Python format-string escaping). The HumanMessage
# templates below still go through ``.format()`` so they retain ``{{`` ``}}``
# escapes around literal JSON braces and positional ``{}`` for substitutions.
# ---------------------------------------------------------------------------

_FACTOR_GUIDELINES_BASE = (
"""
    You are an intelligent and accurate Named Entity Recognition (NER) system with a specialization in Genomics and Biology.\n\n

    I will provide you with GSM XML files (referring to individual samples) and GSE XML files (referring to series of GSM samples). \n\n

    In the XML files provided, your task is to identify the Official Gene Symbol being referenced in the experiment. Specifically, you need to find the following:\n
        1). FACTOR: The Official Gene Symbol of the factor (the target protein that ChIP-seq was conducted on) whose binding sites are being mapped on the genome,
        or in the case of histone post-translational modifications, an abbreviated format. If the experiment does not target a factor, write "None".
        The Factor is generally found after the cell line ontology in the title or in the chip antibody section.
        A Factor is NOT a cell line, cell, species, or anything else besides a transcription factor gene.\n\n

        If the information is not clear from the GSM XML file, you should refer to the corresponding GSE XML file (series document) for additional verification and information.\n
        1). Output Format: Your output must always be a JSON object, structured as follows: \n
            {\n
                "factor": "target binding protein/factor identified",\n
                "epitope_tagged": true_or_false,\n
                "reasoning": "evidence on why this was selected as the target protein for this ChIP-seq experiment"\n
            }\n
            If no factor is identified, the factor should be "None", but the reasoning must still explain why "None" was selected.\n
        2). Roman Numerals: Convert all roman numeral representations into their corresponding numbers. For example, "Pol II" should be converted to "Pol 2".\n
        3). Official Gene Symbol: All factors produced should be in their Official Gene Symbol used by the NCBI, e.g., "ER", "PLXNB3", "TRF-GAA4-1", "H3K27ac". Not in this form, eg., “RNA polymerase II (920102, Biolegend)”, “estrogen receptor.\n
        4). For Post Translational Histone Modifications leave them in their full format do not simplify them down. eg. "H3K27ac" not "H3". "H3" is incorrect. Also remove punctuation from them eg: "H1.4K34ac" should be converted to "H14K34ac" \n
        5). Epitope tag detection: If the chip antibody is against an epitope tag (HA, FLAG, Myc, V5, GFP, mCherry, T7, AU1, AU5, OLLAS, His, Strep, Spot, biotin, BirA, SNAP, HALO), set "epitope_tagged": true. Search the title, source name, growth protocol, treatment protocol, description, and series metadata for the underlying tagged target — look for patterns like "X-tag-Y", "Y-X", "tagged Y", "Y fused to X", "DOX-Y-tag-X", where X is the tag and Y is the underlying biological target. Report Y as the factor. If the antibody is against an epitope tag AND no underlying target can be identified anywhere in the metadata, set "factor": "None" and "epitope_tagged": true. For all non-epitope-tag cases, set "epitope_tagged": false.\n
        6). Empty List: If no entities are presented in any categories, return "None" for the factor but provide reasoning for why "None" was selected.
        \n \n
        Example 1:\n
        Input Format: \n
            \n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM631489 \n
                Status(database="GEO"): \n
                Title: [E-MTAB-223] full_ER_ChIP_T47D_exp1_lane1 \n
                Accession(database="GEO"): GSM631489 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Description: Performer: CRUK-CRI \n
                Channel: \n
                Source: ER_T47D_CRI01 \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="material type"): cell_line \n
                Characteristics(tag="cellline"): T74-D \n
                Characteristics(tag="Sex"): female \n
                Characteristics(tag="diseasestate"): breast cancer \n
                Characteristics(tag="chip antibody"): ER \n
                Growth-Protocol: grow | RPMI 1640 medium por DMEM supplemented with 10% inactivated FBS, l-glutamine and PEST at 37C with 5% CO2. \n
                Molecule: genomic DNA \n
                \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: [E-MTAB-223] ChIP-seq for FOXA1, ER and CTCF in breast cancer cell lines \n
                Accession(database="GEO"): GSE25710 \n
                Summary: Growth cells and map of ER, FoxA1 and CTCF binding at whole genome level. \n

                ArrayExpress Release Date: 2010-10-29 \n

                Person Roles: submitter \n
                Person Last Name: Hurtado \n
                Person First Name: Antoni \n
                Person Email: toni.hurtado@cancer.org.uk \n
                Person Affiliation: Uppsala University \n
                Overall-Design: Experimental Design: high_throughput_sequencing_design \n
                Experimental Design: binding_site_identification_design \n
                Experimental Factor Name: IMMUNOPRECIPITATE \n
                Experimental Factor Name: CELL_LINE \n
                Experimental Factor Type: immunoprecipitate \n
                Experimental Factor Type: cell_line \n
                Type: Genome binding/occupancy profiling by high throughput sequencing \n
            ''' \n
        \n
        Sample Output: \n
            {\n
                "factor": "ER",\n
                "epitope_tagged": false,\n
                "reasoning": "The chip antibody is explicitly listed as 'ER' in the sample metadata, and the series metadata confirms that ER is one of the factors being studied in this ChIP-seq experiment." \n
            }\n

        Example 2:\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM3395078 \n
                Status(database="GEO"): \n
                Title: T47D_POL2_noSerum_ChIP-seq \n
                Accession(database="GEO"): GSM3395078 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Description: ChIP-seq single end (SE) \n
                Channel: \n
                Source: T47D \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="cell line"): T47D \n
                Characteristics(tag="media"): RPMI supplemented with 10% charcoal-treated (CT) FBS for 48 h and starvation was achieved by culturing cells in the absence of FBS for 16 h \n
                Characteristics(tag="serum"): serum-starved \n
                Characteristics(tag="chip-antibody"): rabbit polyclonal antibody anti-Pol II Santa Cruz (N20) (sc-899) \n
                Molecule: genomic DNA \n

            \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: TFIIIC binding to Alu elements controls gene expression via chromatin looping and histone acetylation \n
                Accession(database="GEO"): GSE120162 \n
                Pubmed-ID: 31759822 \n
                Summary: How repetitive elements, epigenetic modifications and architectural proteins interact ensuring proper genome expression remains poorly understood. Here we report regulatory mechanisms unveiling a central role of Alu elements (AEs) and RNA polymerase III transcription factor C (TFIIIC) in structurally and functionally modulating the genome via chromatin looping and histone acetylation. Upon serum deprivation, a subset of AEs pre-marked by the Activity-Dependent Neuroprotector Homeobox protein (ADNP) and located near cell cycle genes recruits TFIIIC, which alters their chromatin accessibility by direct acetylation of histone H3 lysine-18 (H3K18). This facilitates the contacts of AEs with distant CTCF sites near promoter of other cell cycle genes, which also become hyperacetylated at H3K18. These changes ensure basal transcription of cell cycle genes and are critical for their re-activation upon serum re-exposure. Our study reveals how direct manipulation of the epigenetic state of AEs by a general transcription factor regulates 3D genome folding and expression. \n
                Overall-Design: Examination of TFIIIC binding and action during cellular Serum Starvation (SS) \n
                Type: Other \n
            ''' \n
        Sample Output: \n
            {\n
                "factor": "POL2",\n
                "epitope_tagged": false,\n
                "reasoning": "The chip antibody is listed as 'anti-Pol II', which refers to RNA polymerase II. The official gene symbol for RNA polymerase II is 'POL2'."\n
            }\n\n

        Example 2b (Epitope-tagged ChIP):\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM6616013 \n
                Title: ChIP_HA_GI-MEN_ASCL1_HA_tag \n
                Accession(database="GEO"): GSM6616013 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Description: GIMEN_ASCL1 \n
                Channel: \n
                Source: GI-MEN cells \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="cell line"): GI-MEN cells \n
                Characteristics(tag="cell type"): Neuroblastoma \n
                Characteristics(tag="chip antibody"): HA \n
                Growth-Protocol: GI-MEN DOX-ASCL1-tag-HA cells were cultured in RPMI-1640 supplemented with 10% FBS. \n
                Molecule: genomic DNA \n
            ''' \n
        Sample Output: \n
            {\n
                "factor": "ASCL1",\n
                "epitope_tagged": true,\n
                "reasoning": "The chip antibody is HA, which is an epitope tag. The title 'ChIP_HA_GI-MEN_ASCL1_HA_tag', growth protocol 'DOX-ASCL1-tag-HA cells', and description 'GIMEN_ASCL1' all converge on ASCL1 as the underlying tagged target."\n
            }\n\n

        Example 3:\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM4480 \n
                Status(database="GEO"):  \n
                Title: GAPDH \n
                Accession(database="GEO"): GSM4480 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Channel: \n
                Source: cell line \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="cell line"): HeLa \n
                Molecule: genomic DNA \n
            \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: 	Comparing genome-wide chromatin profiles using ChIP-chip or ChIP-seq \n
                Accession(database="GEO"): GSE179 \n
                Pubmed-ID: 202080 \n
                Summary: The goal of the ChIP-seq study was to investigate the distribution of the TATA-binding protein (TBP) across the human genome. TBP is the DNA-binding subunit of the basal transcription factor TFIID for RNA polymerase II (pol II) and it also participates in other complexes for the other RNA polymerase. The BTAF1 ATPase forms a stable complex with TBP and regulates its activity in pol II transcription. BTAF1 is believed to mobilize TBP from promoter and non-promoter sites. To test this hypothesis, TBP ChIP samples were prepared from human HeLa cervix carcinoma cells after knock-down of BTAF1 expression and compared to HeLa cells with a control knock-down of GAPDH. GAPDH is a cytosolic enzyme that participates in glycolysis, and its inactivation is not expected to affect the genomic distribution of TBP, and acts as negative control. ChIP samples were sequenced using SOLiD technology along with the INPUT sample to normalize the distribution of background signals within each of the two chromatin samples. \n
                Overall-Design: 2 ChIP samples + one input sample \n
                Type: Genome binding/occupancy profiling by high throughput sequencing \n
            ''' \n
        Sample Output: \n
            {\n
                "factor": "GAPDH",\n
                "epitope_tagged": false,\n
                "reasoning": "'GAPDH' is the title of the sample. The series data confirms that 'GAPDH' is a target protein of interest."\n
            }\n

        Example 4:\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM6869 \n
                Status(database="GEO"): \n
                Title: LNCaP-H3K4me2-vehicle-siCTRL-Mnase-ChIP-Seq \n
                Accession(database="GEO"): GSM6869 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Description: Chromatin IP against H3K4me2 mononucleosomes in LNCaP cells treated with control siRNA and with vehicle for 4 hrs \n
                Channel: \n
                Source: Prostate cancer cell line (LNCaP) \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="cell line"): LNCaP \n
                Characteristics(tag="sirna transfection"): siCTRL (1027280) \n
                Characteristics(tag="agent"): vehicle \n
                Characteristics(tag="mnase digestion"): yes \n
                Characteristics(tag="chip antibody"): H3K4me2 \n
                Characteristics(tag="chip antibody vendor"): Upstate \n
                Characteristics(tag="chip antibody catalog#"): 07-030 \n
                Characteristics(tag="transgenes"): none \n
                Treatment-Protocol: LNCaP cells were cultured in RPMI 1640 supplemented with 10% FBS.  Control (1027280) and the specific siRNA against  FOXA1 (M-010319) were purchased from Qiagen or Dharmacon. One day prior to transfection, LNCaP cells were seeded in RPMI 1640 medium. Six hours after transfection with Lipofectamine 2000 (Invitrogen), cells were washed twice with PBS and then maintained in hormone-deprived phenol-free RPMI 1640 media.  Cells were then cultured for 96 hours following transfection and then treated with DHT or vehicle for 4 hrs. \n
                Molecule: genomic DNA \n
                \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: Reprogramming Transcriptional Responses through Functionally-Distinct Classes of Enhancers in Prostate Cancer Cells [ChIP-Seq, Gro-Seq] \n
                Accession(database="GEO"): GSE27823 \n
                Pubmed-ID: 21572438 \n
                Summary: Mammalian genomes are populated with thousands of transcriptional enhancers that orchestrate cell type-specific gene expression programs; however, the potential that there are pre-established enhancers in different functional classes that permit alternative signal-dependent transcriptional responses has remained unexplored. Here we present evidence that cell lineage-specific factors, such as FoxA1, can simultaneously facilitate and restrict key regulated transcription factors, exemplified by the androgen receptor (AR), acting at structurally- and functionally-distinct classes of pre-established enhancers, thus licensing specific signal-activated responses while restricting others. Consequently, FoxA1 down-regulation, an unfavorable prognostic sign in advanced prostate tumors, causes a massive switch in AR binding from one functional class of enhancers to another, with a parallel switch in levels of enhancer-templated non-coding RNAs (eRNAs) revealed by the global run-on assay (GRO-seq), which documents the dramatic reprogramming of the hormonal response.  The molecular basis for this switch lies in the release of FoxA1-mediated restriction of AR binding to the new enhancer class with no apparent nucleosome remodeling, which is required for stimulating their eRNA transcription and/or enhancing enhancer:promoter looping and gene activation. Together, these findings reveal a large repository of pre-determined enhancers in the human genome that can be dynamically tuned to induce their transcription and activation of alternative gene expression programs, which may underlie many sequential gene expression events in development or during disease progression. \n
                Overall-Design: ChIP-Seq, Gro-Seq, and gene expression profiling was performed in LNCaP cells with hormone treatment and siRNA against FoxA1 \n
                ChIP-Seq and Gro-Seq data presented here. Supplementary file GroSeq.denovo.transcripts.hg18.bed represents analysis using GSM686948-GSM686950. \n
                Type: Expression profiling by high throughput sequencing \n
                \n
                Title: Reprogramming Transcriptional Responses through Functionally-Distinct Classes of Enhancers in Prostate Cancer Cells \n
                Accession(database="GEO"): GSE27824 \n
                Pubmed-ID: 21572438 \n
                Summary: This SuperSeries is composed of the SubSeries listed below. \n
                Overall-Design: Refer to individual Series \n
                Type: Expression profiling by high throughput sequencing \n
                \n
                ''' \n
            Sample Output: \n
            {\n
                "factor": "H3K4me2",\n
                "epitope_tagged": false,\n
                "reasoning": "The chip antibody is explicitly listed as 'H3K4me2' in the sample metadata, indicating that the experiment targets this histone modification."\n
            }\n\n


        REMINDER! Your output must always be a JSON object, structured as follows: \n
            {\n
                "factor": "target binding protein/factor identified",\n
                "epitope_tagged": true_or_false,\n
                "reasoning": "evidence on why this was selected as the target protein for this ChIP-seq experiment"\n
            }\n
        """
)

_FACTOR_INPUT_TEMPLATE = (
"""
    Please only return the factor that the GSM metadata is referencing use the series metadata to validate your answer. USE only the official gene Symbol used by NCBI as seen in the examples above.
    The official gene symbol is a collection of Letters and Numbers, not words, eg. ESR1, ZFX, RAD51, POL2. ONLY RETURN THE GENE SYMBOL
    \n
    The output should only follow the output format. THE FACTOR SECTION SHOULD NOT HAVE ANY ADDITIONAL WORDS BESIDES THE OFFICIAL GENE NAME FOR THE TARGET PROTEIN. \n
    Output Format:
        {{\n
            "factor": "target binding protein/factor identified",\n
            "epitope_tagged": true_or_false,\n
            "reasoning": "evidence on why this was selected as the target protein for this ChIP-seq experiment"\n
        }}\n
    \n
    Please extract the factor or target protein from this sample:\n
        {}
    \n
    The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
        {}
    \n
    """
)

_FACTOR_RECHECK_INPUT_TEMPLATE = (
"""
    Please only return the factor that the GSM metadata is referencing use the series metadata to validate your answer. USE only the official gene Symbol as seen in the examples above.
    The official gene symbol is a collection of Letters and Numbers, not words, eg. ESR1, ZFX, RAD51, POL2. ONLY RETURN THE GENE SYMBOL
    \n
    NOTE: The factors extracted in previous attempts did NOT match any of the verification checks (gene dataset, transcription factor, chromatin remodeler, histone modification, viral factor, or gene-editing tool). Please reanalyze the metadata and identify the correct Official Gene Symbol. The factor identified this time MUST NOT match any factor in the "Previously Identified Incorrect Factors" list below.\n
    \n
    Previously Identified Incorrect Factors: {}\n
    \n
    The output should only follow the output format. THE FACTOR SECTION SHOULD NOT HAVE ANY ADDITIONAL WORDS BESIDES THE OFFICIAL GENE NAME FOR THE TARGET PROTEIN. \n
    Output Format:
        {{\n
            "factor": "target binding protein/factor identified",\n
            "epitope_tagged": true_or_false,\n
            "reasoning": "evidence on why this was selected as the target protein for this ChIP-seq experiment"\n
        }}\n
    \n
    Please extract the factor or target protein from this sample, ensuring the result is NOT one of the previously identified incorrect factors above:\n
        {}
    \n
    The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
        {}
    \n
    """
)

_ONTOLOGY_GUIDELINES = (
"""
    You are an intelligent and accurate Named Entity Recognition (NER) system with a specialization in Genomics and Biology.\n\n

    I will provide you with GSM XML files (referring to individual samples) and GSE XML files (referring to series of GSM samples). \n\n

    In the XML files provided, your task is to identify the Official Gene Symbol being referenced in the experiment. Specifically, you need to find the following:\n
        1). CELL LINE: The Official Cell Line Symbol of the cell line that ChIP-seq was conducted on, it needs to be in the same format that is found in the Cellosaurus database.\n
        2). CELL TYPE: The official Cell Type of the cell used in the experiment. This should describe the biological characteristics and classification of the cell, such as 'T-cell', 'fibroblast', or 'neuron'.\n
        3). TISSUE: The offical tissue ontology that the ChIP-seq experiment refers to, this describes the anatomical source or origin of the cells, validated against the Uberon and Experiment Factor Ontology. This should describe the tissue type, such as 'lung', 'brain', or 'liver', from which the cells were derived.\n
        4). DISEASE: The offical disease ontology that the ChIP-seq experiment refers to, this is the disease or pathological condition associated with the cell line or tissue.

        Guidelines for Ontologies \n
            1). ONLY Use Official Ontologies provided by Cellosaurus, Experimental Factor Ontology, and Uberon. Also following their naming conventions.\n
            2). Names should NOT be placed in parentheses. If a name is in parentheses, choose the one that best fits the category. For example, 'breast cancer (adenocarcinoma)' should become 'adenocarcinoma'.\n
            3). If the ChIP-seq experiment does not contain information on the given field, use N/A in place.\n
        \n \n
        Example 1:\n
        Input Format: \n
            \n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM631483 \n
                Status(database="GEO"): \n
                Title: [E-MTAB-223] full_ER_ChIP_MCF7_exp2_lane1 \n
                Accession(database="GEO"): GSM631483 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Channel: \n
                Source: JC29_MCF7_ER_full_media_rep2_CRI01 \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="material type"): cell_line \n
                Characteristics(tag="cellline"): MCF-7 \n
                Characteristics(tag="Sex"): female \n
                Characteristics(tag="diseasestate"): breast cancer \n
                Characteristics(tag="chip antibody"): ER \n
                Growth-Protocol: grow | RPMI 1640 medium por DMEM supplemented with 10% inactivated FBS, l-glutamine and PEST at 37C with 5% CO2. \n
                Molecule: genomic DNA \n
                Performer: CRUK-CRI
                \n
                \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: [E-MTAB-223] ChIP-seq for FOXA1, ER and CTCF in breast cancer cell lines \n
                Accession(database="GEO"): GSE25710 \n
                Summary: Growth cells and map of ER, FoxA1 and CTCF binding at whole genome level. \n
                ArrayExpress Release Date: 2010-10-29 \n
                Person Roles: submitter \n
                Person Last Name: Hurtado \n
                Person First Name: Antoni \n
                Person Email: toni.hurtado@cancer.org.uk \n
                Person Affiliation: Uppsala University \n
                Overall-Design: Experimental Design: high_throughput_sequencing_design \n
                Experimental Design: binding_site_identification_design \n
                Experimental Factor Name: IMMUNOPRECIPITATE \n
                Experimental Factor Name: CELL_LINE \n
                Experimental Factor Type: immunoprecipitate \n
                Experimental Factor Type: cell_line \n
                Type: Genome binding/occupancy profiling by high throughput sequencing \n
            ''' \n
        \n
        Sample Output: \n
        {\n
            cell_line: "MCF7", \n
            cell_type: "N/A", \n
            tissue: "breast",\n
            disease: "breast cancer"\n
        }\n\n

        Example 2:\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM3486913 \n
                Status(database="GEO"): \n
                Title: T47D_EV_aHA_rep2 \n
                Accession(database="GEO"): GSM3486913 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Channel: \n
                Source: T47D_EV_aHA \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="cell line"): T47D \n
                Characteristics(tag="cell type"): Breast cancer cell line \n
                Characteristics(tag="chip antibody"): aHA (05-904, Millipore) \n
                Characteristics(tag="plasmid"): control plasmid \n
                Molecule: genomic DNA \n
            \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: Cistromic re-programming by truncating GATA3 mutations promotes mesenchymal transformation in vitro, but not mammary tumour formation in mice [ChIP-seq] \n
                Accession(database="GEO"): GSE122847 \n
                Pubmed-ID: 31218575 \n
                Summary: Heterozygous mutations in the transcription factor GATA3 are identified in 10-15% of all breast cancer cases. Most of these are protein-truncating mutations, concentrated within or downstream of the second GATA-type zinc-finger domain. Here, we investigated the functional consequences of expression of two truncated GATA3 mutants, in vitro in breast cancer cell lines and in vivo in the mouse mammary gland. We found that the truncated GATA3 mutants display altered DNA binding activity caused by preferred tethering through FOXA1. In addition, expression of the truncated GATA3 mutants reduces E-cadherin expression and promotes anchorage-independent growth in vitro. However, we could not identify any effects of truncated GATA3 expression on mammary gland development or mammary tumor formation in mice. Together, our results demonstrate that both truncated GATA3 mutants promote cistromic re-programming of GATA3 in vitro, but these mutants are not sufficient to induce tumor formation in mice. \n
                Overall-Design: Binding of HA-tagged wild-type GATA3 (HA_GATA3_wt) and two truncated variants (HA_GATA3_TR1 and HA_GATA3_TR2) exogenously introduced in T47D cells profiled by ChIP-seq (Chromatin Immunoprecipitation followed by deep sequencing). \n
                Type: Genome binding/occupancy profiling by high throughput sequencing \n
            ''' \n
        \n
        Sample Output: \n
        {\n
            cell_line: "T47D", \n
            cell_type: "N/A", \n
            tissue: "breast",\n
            disease: "breast cancer"\n
        }\n\n

        Example 3:\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM5048518 \n
                Status(database="GEO"): \n
                Title: ChIP-Seq Healthy control sample 10, Alveolar macrophage, Mycobacterium tuberculosis non-challenged library \n
                Accession(database="GEO"): GSM5048518 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Channel: \n
                Source: bronchoalveolar lavage \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="ethnicity"): Caucasian \n
                Characteristics(tag="Sex"): M \n
                Characteristics(tag="age"): 29 \n
                Characteristics(tag="medical.history"): None \n
                Characteristics(tag="sexually transmitted infections"): no \n
                Characteristics(tag="quantiferon tb reagent"): no \n
                Characteristics(tag="cd4 count"): 1091 \n
                Characteristics(tag="prescription drugs"): none \n
                Characteristics(tag="active ingredient"): none \n
                Characteristics(tag="art start year"): none \n
                Characteristics(tag="recreational drugs"): no \n
                Characteristics(tag="cigarette smoker"): no \n
                Characteristics(tag="bal sampling date"): 2017/01/11 \n
                Characteristics(tag="mtb challenged"): no \n
                Characteristics(tag="fragments in clean bam"): 1 \n
                Molecule: genomic DNA \n
                Description: Healthy control sample 10, Alveolar macrophage, Mycobacterium tuberculosis non-challenged library \n
                \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: Epigenetic impairment and blunted transcriptional response to Mycobacterium tuberculosis of alveolar macrophages from persons living with HIV (ChIP-Seq) \n
                Accession(database="GEO"): GSE165705 \n
                Pubmed-ID: 34473646 \n
                Summary: Persons living with HIV (PLWH) are at increased risk of tuberculosis (TB). HIV-associated TB is mainly the result of recent infection with Mycobacterium tuberculosis (Mtb) followed by rapid progression to disease. Alveolar macrophages (AM) are the first cells of the innate immune system that engage Mtb, but how HIV and antiretroviral therapy (ART) impact on the anti-mycobacterial response of AM is not known. In this study AM were challenged in vitro with Mtb, and their epigenetic and transcriptomic responses were determined for PLWH receiving ART, control subjects who were HIV-free (HC) and subjects who received pre-exposure prophylaxis (PrEP) with ART to prevent HIV infection. Compared to HC subjects’ response to Mtb, we showed that AM isolated from PLWH and PrEP subjects displayed substantially weaker transcriptomic response and no significant changes in their chromatin state. These findings revealed a previously unknown adverse effect of ART. \n
                Overall-Design: AM cells were challenged with Mtb at a multiplicity of infection (MOI) of 5:1 or kept in sterile medium for 18-20hrs. Cells were then processed to perform RNA-seq, ATAC-seq and ChIP-seq \n
                Type: Genome binding/occupancy profiling by high throughput sequencing \n
            '''
            \n
        Sample Output: \n
        {\n
            cell_line: "N/A", \n
            cell_type: "alveolar macrophages", \n
            tissue: "lung",\n
            disease: "N/A"\n
        }\n\n

        Example 4:\n
        Input Format:\n
            ''' \n
            This is metadata related to the specific sample:\n
                GSM: GSM6869 \n
                Status(database="GEO"): \n
                Title: LNCaP-H3K4me2-vehicle-siCTRL-Mnase-ChIP-Seq \n
                Accession(database="GEO"): GSM6869 \n
                Type: SRA \n
                Channel-Count: 1 \n
                Description: Chromatin IP against H3K4me2 mononucleosomes in LNCaP cells treated with control siRNA and with vehicle for 4 hrs \n
                Channel: \n
                Source: Prostate cancer cell line (LNCaP) \n
                Organism(taxid="9606"): Homo sapiens \n
                Characteristics(tag="cell line"): LNCaP \n
                Characteristics(tag="sirna transfection"): siCTRL (1027280) \n
                Characteristics(tag="agent"): vehicle \n
                Characteristics(tag="mnase digestion"): yes \n
                Characteristics(tag="chip antibody"): H3K4me2 \n
                Characteristics(tag="chip antibody vendor"): Upstate \n
                Characteristics(tag="chip antibody catalog#"): 07-030 \n
                Characteristics(tag="transgenes"): none \n
                Treatment-Protocol: LNCaP cells were cultured in RPMI 1640 supplemented with 10% FBS.  Control (1027280) and the specific siRNA against  FOXA1 (M-010319) were purchased from Qiagen or Dharmacon. One day prior to transfection, LNCaP cells were seeded in RPMI 1640 medium. Six hours after transfection with Lipofectamine 2000 (Invitrogen), cells were washed twice with PBS and then maintained in hormone-deprived phenol-free RPMI 1640 media.  Cells were then cultured for 96 hours following transfection and then treated with DHT or vehicle for 4 hrs. \n
                Molecule: genomic DNA \n
                \n
            The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
                Title: Reprogramming Transcriptional Responses through Functionally-Distinct Classes of Enhancers in Prostate Cancer Cells [ChIP-Seq, Gro-Seq] \n
                Accession(database="GEO"): GSE27823 \n
                Pubmed-ID: 21572438 \n
                Summary: Mammalian genomes are populated with thousands of transcriptional enhancers that orchestrate cell type-specific gene expression programs; however, the potential that there are pre-established enhancers in different functional classes that permit alternative signal-dependent transcriptional responses has remained unexplored. Here we present evidence that cell lineage-specific factors, such as FoxA1, can simultaneously facilitate and restrict key regulated transcription factors, exemplified by the androgen receptor (AR), acting at structurally- and functionally-distinct classes of pre-established enhancers, thus licensing specific signal-activated responses while restricting others. Consequently, FoxA1 down-regulation, an unfavorable prognostic sign in advanced prostate tumors, causes a massive switch in AR binding from one functional class of enhancers to another, with a parallel switch in levels of enhancer-templated non-coding RNAs (eRNAs) revealed by the global run-on assay (GRO-seq), which documents the dramatic reprogramming of the hormonal response.  The molecular basis for this switch lies in the release of FoxA1-mediated restriction of AR binding to the new enhancer class with no apparent nucleosome remodeling, which is required for stimulating their eRNA transcription and/or enhancing enhancer:promoter looping and gene activation. Together, these findings reveal a large repository of pre-determined enhancers in the human genome that can be dynamically tuned to induce their transcription and activation of alternative gene expression programs, which may underlie many sequential gene expression events in development or during disease progression. \n
                Overall-Design: ChIP-Seq, Gro-Seq, and gene expression profiling was performed in LNCaP cells with hormone treatment and siRNA against FoxA1 \n
                ChIP-Seq and Gro-Seq data presented here. Supplementary file GroSeq.denovo.transcripts.hg18.bed represents analysis using GSM686948-GSM686950. \n
                Type: Expression profiling by high throughput sequencing \n
                \n
                Title: Reprogramming Transcriptional Responses through Functionally-Distinct Classes of Enhancers in Prostate Cancer Cells \n
                Accession(database="GEO"): GSE27824 \n
                Pubmed-ID: 21572438 \n
                Summary: This SuperSeries is composed of the SubSeries listed below. \n
                Overall-Design: Refer to individual Series \n
                Type: Expression profiling by high throughput sequencing \n
                \n
                ''' \n
            Sample Output: \n
            {\n
                cell_line: "LNCaP", \n
                cell_type: "prostate cancer cell", \n
                tissue: "prostate",\n
                disease: "prostate cancer"\n
            }\n\n
        """
)

_ONTOLOGY_INPUT_TEMPLATE = (
"""
    You are to identify the cell line, cell type, tissue, and disease of the cell that the ChIP-seq experiment was performed on. \n
    Please return the officical symbol names for the ontologies, please use the naming conventions from Cellosaurus, Experimental Factor Ontology, and Uberon. \n
    Extract the following information from the ChIP-seq experiment metadata:\n
        1). CELL LINE: The official Cell Line Symbol of the cell line that ChIP-seq was conducted on. \n
        2). CELL TYPE: The official Cell Type of the cell used in the experiment. \n
        3). TISSUE: The offical tissue ontology that the ChIP-seq experiment refers to, this describes the anatomical source or origin of the cells.\n
        4). DISEASE: The offical disease ontology that the ChIP-seq experiment refers to. \n

    \n
    Please extract the ontologies from this sample:\n
        {}
    \n
    The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
        {}
    \n
    """
)


### General Functionality ###
def get_data_dir():
    """Returns the path to the data directory within the installed package."""
    return Path(__file__).parent / "data"

def clean_input(input_string):
    """
    Cleans the input string by converting it to lowercase and removing spaces, 
    newlines, and non-alphanumeric characters. Returns an empty string if input is None.

    Args:
        input_string (str): The string to be cleaned.

    Returns:
        str: The cleaned string.
    """
    if input_string is None:
        return ''
    # Convert the input string to lowercase, remove spaces, newlines, and symbols
    cleaned_string = re.sub(r'[^\w]', '', input_string.lower())
    return cleaned_string

def clean_input_fuzzy(input_string):
    """
    Cleans the input string by converting it to lowercase and removing symbols,
    but preserving spaces.
    """
    if input_string is None:
        return ''
    return re.sub(r'[^\w\s]', '', input_string.lower())

def remove_words(text):
    """
    Removes predefined words from the input text, specifically for ontology matching.
    To ensure that LLM extracted ontologies match important words.

    Args:
        text (str): The input text from which words will be removed.

    Returns:
        str: The text with specified words removed.
    """
    words_to_remove = ["cell", "line", "primary", "the", "of", "and", 
                       "or", "but", "tissue", "cells", "human", "for", 
                       "organ", "region", "system", "sample", "assay", 
                       "disease", "to","measurement", "down", "up", "part",
                       "layer", "membrane", "area", "to", "like", "related",
                       "type", "with", "amount", "containing"]
    for word in words_to_remove:
        text = text.replace(word, "")
    return text.strip()

def process_string(s, remove=False):
        if not s:  # Handles None and empty string cases
            return None
        if remove:
            words_to_remove = ["cell", "line", "primary", "the", "of", "and", 
                           "or", "but", "tissue", "cells", "human", "for", 
                           "organ", "region", "system", "sample", "assay", 
                           "disease", "to","measurement", "down", "up", "part",
                           "layer", "membrane", "area", "to", "like", "related",
                           "type", "with", "amount", "containing"]
        else:
            words_to_remove = []
        
        words = s.split()
        cleaned_words = [clean_input(word) for word in words]
        filtered_words = [word for word in cleaned_words if word not in words_to_remove and word != '']
        return "".join(filtered_words)

def remove_invalid_characters_from_file(file_path):
    """
    Removes invalid characters from an XML file and ensures the XML is well-formed. 
    Returns the cleaned XML as a string if valid, otherwise returns None.

    Args:
        file_path (str): The path to the XML file.

    Returns:
        str or None: The cleaned XML string if well-formed, otherwise None.
    """
    # Read XML from file
    try:
        with open(file_path, 'r', encoding='iso-8859-1') as file:
            xml_data = file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except UnicodeDecodeError:
        print(f"Error decoding file {file_path}. Try specifying the correct encoding.")
        return None

    # Define a regular expression to match invalid characters
    invalid_char_regex = re.compile(
        '[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]')

    # Remove invalid characters using the regex
    cleaned_xml_string = invalid_char_regex.sub('', xml_data)

    # Parse the cleaned XML string to check for XML well-formedness
    try:
        root = ET.fromstring(cleaned_xml_string)
        cleaned_xml_string = ET.tostring(root, encoding='unicode', method='xml')
        return cleaned_xml_string
    except ET.ParseError:
        print("Invalid XML structure after removing invalid characters.")
        return None

def collapse_ontology_terms(entries):
    """Collapse a list of ontology match dicts, removing duplicates and merging
    entries that share the same (official_term, ontology_accession) pair."""

    def _make_hashable(v):
        """Convert a value to a hashable form for deduplication."""
        if isinstance(v, list):
            return tuple(sorted(str(i) for i in v))
        return v

    # --- Step 1: Remove fully identical entries ---
    seen = set()
    unique_entries = []
    for entry in entries:
        fingerprint = tuple(sorted(
            (k, _make_hashable(v)) for k, v in entry.items()
        ))
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_entries.append(entry)

    # --- Step 2: Group by (official_term, ontology_accession) ---
    grouped = {}
    for entry in unique_entries:
        group_key = (entry.get('official_term', ''), entry.get('ontology_accession', ''))
        grouped.setdefault(group_key, []).append(entry)

    collapsed_entries = []

    for (official_term, accession), group in grouped.items():
        collapsed_entry = {'official_term': official_term}

        # Collect all field keys across the group
        keys = set()
        for item in group:
            keys.update(item.keys())
        keys.discard('official_term')

        for key in keys:
            values = []
            for item in group:
                if key in item and item[key] is not None:
                    values.append(item[key])

            # Deduplicate, handling unhashable types (lists, dicts)
            unique_values = []
            seen_values = set()
            for v in values:
                h = _make_hashable(v)
                if h not in seen_values:
                    seen_values.add(h)
                    unique_values.append(v)

            if len(unique_values) == 1:
                collapsed_entry[key] = unique_values[0]
            elif len(unique_values) > 1:
                collapsed_entry[key] = unique_values

        collapsed_entries.append(collapsed_entry)

    return collapsed_entries

def _load_xml_root(xml_input):
    """
    Accept either a path to an XML file or an XML string, and return the
    ElementTree root. MINiML payloads always start with '<?xml', so a stripped
    input beginning with '<' is treated as inline XML; everything else is
    treated as a file path.
    """
    if isinstance(xml_input, str) and xml_input.lstrip().startswith("<"):
        return ET.fromstring(xml_input)
    return ET.parse(xml_input).getroot()


def simplify_gsm_xml_file(xml_file):
    """
    Simplifies and extracts relevant data from a GSM MINiML XML by excluding
    specified elements and handling "Channel" elements separately. Includes
    attributes and text content for clarity.

    Args:
        xml_file: Path to a GSM XML file, or the XML body itself as a string
            (e.g., as returned by ``geo_fetch.fetch_geo_xml``).

    Returns:
        str: A formatted string containing the extracted data.
    """
    # Define the XML namespace
    ns = {"ns": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}

    # Parse the XML (file path or inline XML string)
    root = _load_xml_root(xml_file)

    # Find the "Sample" element
    sample = root.find("ns:Sample", ns)
    
    # Dictionary to store the extracted data
    extracted_data = {}

    # Elements to exclude globally
    exclude_elements = {"Platform", "Platform-Ref", "Library-Source", 
                        "Library-Selection", "Instrument-Model", "Contact-Ref", 
                        "Supplementary-Data", "Relation", "Data-Processing", 
                        "Extract-Protocol"}
    
    # Custom handling for "Channel" elements
    channel_data = []
    
    for child in sample:
        tag = child.tag.split("}")[-1]  # Remove namespace prefix
        
        # Skip globally excluded tags
        if tag in exclude_elements:
            continue
        
        # Handle "Channel" elements separately
        if tag == "Channel":
            channel_content = []
            for subchild in child:
                subtag = subchild.tag.split("}")[-1]
                # Skip "Extract-Protocol" inside Channel
                if subtag == "Extract-Protocol":
                    continue
                
                # Add attributes (like tag="value") if available
                attributes = " ".join(f'({k}="{v}")' for k, v in subchild.attrib.items())
                value = f"{subtag}{attributes}: {subchild.text.strip()}" if subchild.text else None
                
                if value:
                    channel_content.append(value)
            if channel_content:
                channel_data.append("\n".join(channel_content))
        else:
            # Handle other elements and include attributes if available
            attributes = " ".join(f'({k}="{v}")' for k, v in child.attrib.items())
            value = f"{tag}{attributes}: {child.text.strip()}" if child.text else None
            if value:
                extracted_data[tag] = value

    # Build the final output
    output_lines = []
    
    for key, value in extracted_data.items():
        output_lines.append(value)  # Already formatted with attributes

    if channel_data:
        output_lines.append("Channel:")
        output_lines.append("\n".join(channel_data))

    # Join all lines with newlines
    output = "\n".join(output_lines)
    return output

def simplify_gse_xml_file(xml_file):
    """
    Simplifies and extracts relevant data from a GSE MINiML XML by excluding
    specified elements and including attributes for clarity.

    Args:
        xml_file: Path to a GSE XML file, or the XML body itself as a string
            (e.g., as returned by ``geo_fetch.fetch_geo_xml``).

    Returns:
        str: A formatted string containing the extracted data.
    """
    # Define the XML namespace
    ns = {"ns": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}

    # Parse the XML (file path or inline XML string)
    root = _load_xml_root(xml_file)

    # Find the Series element
    series = root.find("ns:Series", ns)
    if series is None:
        return "Error: <Series> element not found in XML."

    # Elements to exclude
    exclude_elements = {"Status", "Contributor-Ref", "Sample-Ref", "Relation", "Supplementary-Data"}
    
    # Dictionary to store extracted data
    extracted_data = {}

    # Iterate over the Series children
    for child in series:
        tag = child.tag.split("}")[-1]  # Remove namespace prefix

        # Skip excluded tags
        if tag in exclude_elements:
            continue
        
        # Include attributes (e.g., database="GEO")
        attributes = " ".join(f'({k}="{v}")' for k, v in child.attrib.items())
        value = f"{tag}{attributes}: {child.text.strip()}" if child.text else None
        
        if value:
            extracted_data[tag] = value

    # Build the final output
    output_lines = [value for key, value in extracted_data.items()]
    return "\n".join(output_lines)

def _load_json_data(file_path):
    """Helper function to load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file '{file_path}': {e}")
        return None

def _parse_gsm_ids_input(gsm_ids_input):
    """
    Parse GSM IDs input - can be a list/array or a JSON file path.
    Returns a list of GSM ID strings.
    """
    # Check if it's a string (file path)
    if isinstance(gsm_ids_input, str):
        data = _load_json_data(gsm_ids_input)
        if data is None:
            return []
        return data if isinstance(data, list) else []
    
    # Handle various array-like inputs (list, pandas Series, numpy array, etc.)
    try:
        # Convert to list if it's pandas Series, numpy array, etc.
        if hasattr(gsm_ids_input, 'tolist'):
            return gsm_ids_input.tolist()
        elif hasattr(gsm_ids_input, '__iter__') and not isinstance(gsm_ids_input, str):
            return list(gsm_ids_input)
        else:
            return [gsm_ids_input] if gsm_ids_input else []
    except Exception as e:
        print(f"Error parsing GSM IDs input: {e}")
        return []

def _format_output_structure(gsm_id, extracted_factor=None, factor_status=None,
                              factor_type=None, extracted_ontology=None):
    """
    Format the output according to the standardized structure.

    A factor entry now always includes ``extracted_factor`` and ``factor_type``
    (when the latter is supplied), and optionally ``factor_status`` to
    signal a notable success modifier (e.g., ``epitope_tagged``) or a
    failure mode (e.g., ``control_sample``, ``verification_failed``).
    """
    result = {gsm_id: {}}

    if extracted_factor is not None:
        factor_entry = {"extracted_factor": extracted_factor}
        if factor_type is not None:
            factor_entry["factor_type"] = factor_type
        if factor_status:
            factor_entry["factor_status"] = factor_status
        result[gsm_id]["factor"] = factor_entry

    if extracted_ontology is not None:
        # Handle the ontology data - use validated terms if available, otherwise fall back to "N/A"
        result[gsm_id]["ontology"] = {
            "extracted_ontologies": {
                "cell_line": extracted_ontology.get("cell_line") if extracted_ontology.get("cell_line") is not None else "N/A",
                "cell_type": extracted_ontology.get("cell_type") if extracted_ontology.get("cell_type") is not None else "N/A", 
                "tissue": extracted_ontology.get("tissue") if extracted_ontology.get("tissue") is not None else "N/A",
                "disease": extracted_ontology.get("disease") if extracted_ontology.get("disease") is not None else "N/A"
            }
        }
    
    return result

### Factor Extraction ###
def is_control(gsm_xml_string, model=None):
    # Define the instruction and input prompts
    GUIDELINES_PROMPT = (
        """
        You are a highly skilled bioinformatics assistant. Analyze the given metadata for a ChIP-seq experiment and determine if it describes a control or DNase-seq experiment.

        A control experiment is a **separate** experiment that does not use a target protein but instead serves as a baseline for comparison. The control can often be identified in the "Title" or the "ChIP Antibody" section, or the "Library strategy" section and may include any of the following keywords:
        - None
        - Input
        - Control
        - Total Input or TI
        - IgG
        - Mock
        - Bead-only
        - WCE
        - Whole Cell Extract
        - DNase
        - DNase-seq
        - DNase Hypersensitivity
        - ChAP-seq
        - ChAP
        - Ubiquitin

        ### **Instructions:**
        1. **Carefully examine the metadata** to determine whether it describes a control, DNase-seq, or ChAP-seq (uses Ubiquitin) experiment.
        2. If it's a control experiment for the target protein, which may contain any of the keywords above, output "True".
        3. If the experiment uses a control for the cell line or species, output "False". Only output "True" for control experiments concerning the target protein.
        4. Otherwise, output "False".
        5. Be especially scrutinous when control-related terms appear; make sure the experiment itself is a control experiment, not a non-control target experiment that is merely referencing or providing context about the matched control used for comparison.

        PLEASE only output "True" or "False" — do not provide any other information.
        """
    )

    INPUT_PROMPT = (
        """
        Please determine if this ChIP-seq experiment is using a control for the factor/target protein of this experiment or not.

        The output should only follow the output format of either "True" or "False".

        Please determine if this sample is a control:
        {}
        """
    )

    formatted_prompt = INPUT_PROMPT.format(gsm_xml_string)

    setup_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=GUIDELINES_PROMPT),
            HumanMessage(content=formatted_prompt)
        ]
    )

    try:
        chat_message = setup_messages.format_messages()
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        # Extract just "True" or "False" from response, handling models that add extra text
        content = res.content.strip().lower()
        if "true" in content:
            return "True"
        return "False"
    except Exception as e:
        raise e

def extract_factor(gsm_xml_string, gse_xml_strings, model=None):
    """
    Extracts the transcription factor or target protein being mapped in a ChIP-seq experiment from GSM and GSE XML metadata strings.

    Parameters:
        gsm_xml_string (str): String content of the GSM XML file.
        gse_xml_strings (List[str]): A list of string contents for the GSE XML files.

    Returns:
        tuple[str, bool]: A two-tuple ``(factor, epitope_tagged)``.

            * ``factor`` is the JSON-extracted target binding protein string
              (e.g., ``"ER"``, ``"FOXA1"``, ``"H3K27ac"``), or ``"None"`` when no
              target could be identified.
            * ``epitope_tagged`` is ``True`` when the LLM determined that the
              ChIP antibody is against an epitope tag (HA, FLAG, Myc, V5, GFP,
              etc.). When this is ``True`` and ``factor != "None"``, the factor
              is the recovered underlying target. When this is ``True`` and
              ``factor == "None"``, no underlying target was recoverable.

    Notes:
        - Transcription factors are extracted based on specific rules:
            1. Official Gene Symbols (NCBI format) are prioritized.
            2. Roman numerals are converted to numbers (e.g., "Pol II" becomes "Pol 2").
            3. Histone modifications are kept in full form (e.g., "H3K27ac").
        - If the GSM metadata does not provide sufficient information, the function refers to GSE metadata for additional context.
    """

    # Simplify GSM XML file
    xml_prompt = gsm_xml_string
    gse_prompts = "\n\n".join(gse_xml_strings)

    # System prompt is byte-identical across all extract_factor and
    # factor_recheck calls so providers that cache by prefix hash (Anthropic
    # explicitly, OpenAI/Gemini/DeepSeek automatically) share one cache entry.
    formatted_input = _FACTOR_INPUT_TEMPLATE.format(xml_prompt, gse_prompts)
    chat_message = [
        _make_cached_system_message(_FACTOR_GUIDELINES_BASE, model),
        HumanMessage(content=formatted_input),
    ]

    try:
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        _log_cache_usage(res, label="extract_factor")
        res_content = res.content
        return_dict = _parse_llm_json(res_content)
        result = return_dict.get("factor")
        epitope_tagged = bool(return_dict.get("epitope_tagged", False))
        return result, epitope_tagged
    except Exception as e:
        raise

def match_human_gene(df, factor):
    """
    Finds matching genes from a NCBI offical Homo_sapiens.gene_info data set.

    Args:
        df (pd.DataFrame): The dataframe containing gene information. Must include 'Symbol' and 'Synonyms' columns.
        factor (str): The factor to match against genes in the dataframe.

    Returns:
        list: A list of rows where the gene matches.
    """

    # Validate required columns
    required_columns = {"Symbol", "Synonyms"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    cleaned_factor = clean_input(factor)

    # Filter rows where 'Symbol' or 'Synonyms' match the cleaned factor
    def synonym_match(synonyms_str, factor):
        synonyms = map(clean_input, synonyms_str.split("|"))
        return cleaned_factor in synonyms

    matching_rows = df[
        df["Symbol"].apply(clean_input).eq(cleaned_factor) | 
        df["Synonyms"].apply(lambda x: synonym_match(x, cleaned_factor))
    ]

    matching_list = matching_rows.to_dict(orient="records")
    return matching_list

def validate_transcription_factor(matching_factors, TF_DB):
    """
    Validates transcription factors by filtering them against a reference transcription factor database.

    Args:
        matching_factors (list): A list of factors to validate, where each factor is a dictionary containing a 'Symbol' key.
        TF_DB (pd.DataFrame): A DataFrame containing valid transcription factor symbols with a column named 'Symbol'.

    Returns:
        list: A filtered list of transcription factors that match the valid symbols in the TF_DB.
    """
    # Extract valid transcription factor symbols into a set for O(1) lookups
    valid_symbols = set(TF_DB['Symbol'])
    # Filter matching factors based on the valid symbols
    return [factor for factor in matching_factors if factor.get('Symbol') in valid_symbols]

def validate_chromatin_remodelers(matching_factors, CR_db):
    """
    Validates chromatin remodelers by filtering them against a database of valid chromatin remodeling factors and synonyms.

    Args:
        matching_factors (list): A list of factors to validate, where each factor is a dictionary containing a 'Symbol' key.
        chromatin_db (dict): A dictionary with keys 'chromatin remodeling' and 'synonyms', representing valid symbols and their synonyms.

    Returns:
        list: A filtered list of chromatin remodelers that match the valid symbols or their synonyms in the chromatin_db.
    """
    # Initialize valid symbols set
    valid_symbols = set()

    # Add chromatin remodeling symbols to valid_symbols
    for symbol in CR_db.get('chromatin remodeling', []):
        cleaned_symbol = clean_input(symbol)
        valid_symbols.add(cleaned_symbol)

    # Add synonyms to valid_symbols
    for synonyms in CR_db.get('synonyms', []):
        if pd.notna(synonyms):  # Check if synonyms is not NaN
            for synonym in synonyms.split(','):
                cleaned_synonym = clean_input(synonym.strip())
                valid_symbols.add(cleaned_synonym)

    # Filter the matching_factors list
    filtered_factors = [
        factor for factor in matching_factors
        if clean_input(factor.get('Symbol', '')) in valid_symbols
    ]

    return filtered_factors

def _build_factor_index(db):
    """
    Build a flat lookup index from a viral or gene-editing tool DB.

    Returns a dict mapping ``clean_input(synonym)`` to the entry's
    ``official_symbol`` for O(1) case-insensitive matching that also
    tolerates dash/underscore/whitespace noise.

    Args:
        db (dict): Loaded JSON DB with a top-level ``"factors"`` list, where
            each entry has an ``"official_symbol"`` and optional ``"synonyms"``.

    Returns:
        dict[str, str]: Cleaned-key to official-symbol lookup.
    """
    index = {}
    if not isinstance(db, dict):
        return index
    for entry in db.get("factors", []):
        official = entry.get("official_symbol")
        if not official:
            continue
        synonyms = list(entry.get("synonyms", [])) + [official]
        for syn in synonyms:
            key = clean_input(syn)
            if key:
                index[key] = official
    return index


def validate_viral_factor(factor, viral_index):
    """
    Look up a candidate factor against the viral factor index.

    Args:
        factor (str): The candidate factor string (e.g., ``"LANA"``, ``"BZLF1"``).
        viral_index (dict): A lookup dict produced by :func:`_build_factor_index`.

    Returns:
        str | None: The official viral symbol if the factor matches, else ``None``.
    """
    if not factor or factor == "None":
        return None
    if not viral_index:
        return None
    key = clean_input(factor)
    if not key:
        return None
    return viral_index.get(key)


def validate_gene_editing_tool(factor, editing_index):
    """
    Look up a candidate factor against the gene-editing tool index.

    Args:
        factor (str): The candidate factor string (e.g., ``"Cas9"``, ``"dCas9-KRAB"``).
        editing_index (dict): Lookup dict produced by :func:`_build_factor_index`.

    Returns:
        str | None: The official editing tool symbol if matched, else ``None``.
    """
    if not factor or factor == "None":
        return None
    if not editing_index:
        return None
    key = clean_input(factor)
    if not key:
        return None
    return editing_index.get(key)


def classify_human_gene(symbol, TF_df, chromatin_df):
    """
    Classify a verified human gene symbol as TF, chromatin remodeler, or generic gene.

    The classification mirrors the lookup patterns already used by
    :func:`validate_transcription_factor` and
    :func:`validate_chromatin_remodelers` so that downstream behavior is
    consistent. The function never raises — on unexpected DataFrame
    contents it falls back to ``"gene"``.

    Args:
        symbol (str): An official gene symbol (e.g., ``"ESR1"``).
        TF_df (pd.DataFrame): Human TF table with a ``Symbol`` column.
        chromatin_df: Chromatin remodeler table (DataFrame or dict-like).

    Returns:
        str: One of ``"transcription_factor"``, ``"chromatin_remodeler"``,
        or ``"gene"``.
    """
    if not symbol:
        return "gene"

    try:
        if symbol in set(TF_df['Symbol']):
            return "transcription_factor"
    except Exception:
        pass

    cleaned_symbol = clean_input(symbol)

    try:
        for cr_symbol in chromatin_df.get('chromatin remodeling', []) or []:
            if clean_input(cr_symbol) == cleaned_symbol:
                return "chromatin_remodeler"
    except Exception:
        pass

    try:
        for syns in chromatin_df.get('synonyms', []) or []:
            if pd.notna(syns):
                for s in str(syns).split(','):
                    if clean_input(s.strip()) == cleaned_symbol:
                        return "chromatin_remodeler"
    except Exception:
        pass

    # Also try the actual underscore-form column ('chromatin_remodeler') in
    # case the loaded DataFrame keeps the literal CSV header. This lets the
    # classifier work whether the caller passes a DataFrame whose access
    # surface exposes that column name or the dict-style "chromatin remodeling"
    # key used by the legacy validator.
    try:
        if 'chromatin_remodeler' in getattr(chromatin_df, 'columns', []):
            for cr_symbol in chromatin_df['chromatin_remodeler'].dropna().tolist():
                if clean_input(cr_symbol) == cleaned_symbol:
                    return "chromatin_remodeler"
            if 'synonyms' in chromatin_df.columns:
                for syns in chromatin_df['synonyms'].dropna().tolist():
                    for s in str(syns).split(','):
                        if clean_input(s.strip()) == cleaned_symbol:
                            return "chromatin_remodeler"
    except Exception:
        pass

    return "gene"


def validate_histone_mark(mark):
    """
    Validates histone modification strings with support for:
    - Full histone variant names and synonyms
    - Multiple PTMs in a single mark (e.g., H3K27acK36me3)
    - Valid residues for each PTM
    - Optional 's' or 'a' suffix to indicate symmetric/asymmetric
    """
    
    HISTONE_VARIANTS = {
        "H1", "H1.0", "H1.1", "H1.2", "H1.3", "H1.4", "H1.5", "H1.6", "H1.7", "H1ts", "H1FNT", "H1t", "H1.8", "H1oo",
        "H1Foo", "H1.9", "H1.10", "H1X",
        "H2A", "H2A.1", "H2A.2", "H2A.X", "H2A.Z", "mH2A", "macroH2a", "H2A.B", "H2A.L", "H2A.P", "H2A.J", "H2A.Lap1", "H2A.Bbd",
        "H2B", "H2B.1", "H2B.W", "H2B.Z", "H2B.K", "H2B.N", "H2BL1",
        "H3", "H3.1", "H3.2", "H3.3", "H3.4", "H3.5", "H3.X", "H3.Y", "H3.6", "CENPA", "H3.7", "H3.8", "H3T",
        "H4"
    }

    VALID_PTMS = {
        'K': {'ac', 'ar1', 'bio', 'but', 'cr', 'for', 'hib', 'mal', 'me1', 'me2', 'me3', 'oh', 'su', 'ub', 'gl', 'ph', 'gt'},
        'R': {'ar1', 'cit', 'me1', 'me2', 'me3'},
        'S': {'ph', 'og', 'fa', 'pal', 'amp'},
        'T': {'ph', 'og', 'fa', 'amp'},
        'Y': {'ph', 'ox', 'amp', 'sul'},
        'C': {'ar1', 'gt', 'ox', 'fa', 'pal', 'nit', 'pt'},
        'E': {'ar1', 'iso', 'pyr'},
        'D': {'ar1', 'iso'},
        'P': {'oh'},
        'M': {'ox'},
        'W': {'ox', 'ph'},
        'G': {'fa', 'myr'},
        'N': {'fa', 'pal'},
        'Q': {'pyr'}
    }

    SYMMETRY_SUFFIXES = {'s', 'a'}

    # Step 1: extract histone variant
    histone = None
    for variant in sorted(HISTONE_VARIANTS, key=len, reverse=True):
        if mark.startswith(variant):
            histone = variant
            mod_part = mark[len(variant):]
            break

    if not histone:
        return False

    i = 0
    while i < len(mod_part):
        residue = mod_part[i].upper()
        if residue not in VALID_PTMS:
            return False
        i += 1

        # Match digits (position)
        m = re.match(r'\d+', mod_part[i:])
        if not m:
            return False
        pos = m.group()
        i += len(pos)

        # Try to match PTM
        matched = False
        for ptm in sorted(VALID_PTMS[residue], key=len, reverse=True):
            if mod_part[i:].lower().startswith(ptm):
                i += len(ptm)
                matched = True
                break
        if not matched:
            return False

        # Optional symmetry suffix
        if i < len(mod_part) and mod_part[i] in SYMMETRY_SUFFIXES:
            i += 1

    return True

def generate_synonyms(factor, model=None):
    """
    Generates a list of synonyms for a genomic binding protein or gene using a specified LLM (Language Learning Model).

    Args:
        factor (str): The genomic binding protein or gene name for which synonyms are to be generated.

    Returns:
        str: A string containing the generated synonyms in the format: ["synonym1", "synonym2", ..., "synonymX"].

    Behavior:
        - Constructs a formatted prompt to instruct the LLM on generating gene synonyms.
        - Sends the formatted prompt to the LLM and retrieves the generated synonyms.
        - Returns the output as a string of synonyms.
    """
    try:
        # Construct input prompt
        INPUT_PROMPT = (
            """
            You are an intelligent and accurate system with a specialization in Genomics and Biology.
            We are trying to match a genomic binding protein or gene to an official NCBI database for human genes using their official gene ontologies.
            Your job is to receive an input genomic binding protein or gene that was searched for using ChIP-seq and output either other possible names that the gene may match to in this database 
            or the OFFICIAL gene name synonyms of that binding protein or gene.

            Your output should be in this format as follows: ["synonym1", "synonym2", ..., "synonymX"]

            Now find the synonyms of this genomic binding protein or gene: {}
            """
        )

        formatted_prompt = INPUT_PROMPT.format(factor)

        # Prepare the LLM setup
        setup_messages = ChatPromptTemplate.from_messages(
            [HumanMessage(content=formatted_prompt)]
        )
        chat_message = setup_messages.format_messages()
        
        # Initialize and invoke the LLM
        llm = _init_llm(model)
        response = llm.invoke(chat_message)

        # Parse the list from the response (handles markdown fences, preamble, etc.)
        return _parse_llm_list(response.content)

    except Exception as e:
        raise

def factor_recheck(gsm_xml_string, gse_xml_strings, factor_fails, model=None):
    """
    Reanalyzes metadata for a given GSM sample to identify the correct experimental factor while avoiding previously identified incorrect factors.

    This function processes metadata files for a given GSM sample and its associated series, 
    applies predefined guidelines to extract the official gene symbol of the target protein 
    or histone modification being studied, and returns the correct factor in a JSON-compatible format.

    Args:
        gsm_xml_string (str): String content of the GSM XML file.
        gse_xml_strings (List[str]): A list of string contents for the GSE XML files.
        factor_fails (list of str): A list of previously identified incorrect factors to avoid in reanalysis.

    Returns:
        str: JSON-compatible string containing the transcription factor along with reasoning. For enhanced model performance. 
            The format string, e.g., "ER", "FOXA1". If no entities are identified, None.
    Parameters:
        gsm_xml_string (str): String content of the GSM XML file.
        gse_xml_strings (List[str]): A list of string contents for the GSE XML files.

    Notes:
        - Metadata analysis adheres to the guidelines specified for identifying factors, including 
        format standardization, avoiding prior errors, and checking both sample and series metadata.
        - Output format strictly excludes any additional text, explanation, or metadata outside of 
        the JSON-compatible array of strings.

    Example:
        factor_recheck("GSM123456", ["MCF-7", "ESR1"])
        "ZFX"
    """
    # Simplified XML files
    xml_prompt = gsm_xml_string
    gse_prompts = "\n\n".join(gse_xml_strings)

    # Reuse the same system prompt that extract_factor uses so the cached
    # prefix is shared. The "previously rejected factors" context lives in
    # the human message (which is dynamic per-sample anyway).
    formatted_input = _FACTOR_RECHECK_INPUT_TEMPLATE.format(
        ", ".join(factor_fails),
        xml_prompt,
        gse_prompts,
    )
    chat_message = [
        _make_cached_system_message(_FACTOR_GUIDELINES_BASE, model),
        HumanMessage(content=formatted_input),
    ]

    try:
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        _log_cache_usage(res, label="factor_recheck")
        res_content = res.content
        return_dict = _parse_llm_json(res_content)
        result = return_dict.get("factor")
        epitope_tagged = bool(return_dict.get("epitope_tagged", False))
        return result, epitope_tagged
    except Exception as e:
        raise

def tf_picker(TF_factors, gsm_xml_string, gse_xml_strings, model=None):
    """
    Identifies the most likely transcription factor referenced in ChIP-seq metadata.

    This function analyzes GSM and GSE XML files, simplifies their content, and uses 
    a specified LLM to determine which transcription factor from a given list is most 
    likely referenced in the metadata.

    Args:
        TF_factors (list): A list of transcription factors (gene symbols) to consider.
        gsm_xml_string (str): String content of the GSM XML file.
        gse_xml_strings (List[str]): A list of string contents for the GSE XML files.

    Returns:
        str: The gene symbol of the identified transcription factor.
    """

    TF_factors = [str(d) for d in TF_factors]

    # Simplify GSM XML file
    xml_prompt = gsm_xml_string
    gse_prompts = "\n\n".join(gse_xml_strings)

    GUIDELINES_PROMPT = (
        """
        You are an intelligent and accurate Named Entity Recognition (NER) system with a specialization in Genomics and Biology.\n\n

        I will provide you with GSM XML files (referring to individual samples) and GSE XML files (referring to series of GSM samples). \n
        I will also provide you with a list of transcription factors that could be referenced in the GSM and GSE XML Files \n\n

        In the XML files provided, your task is to identify which transcription factor from the list of transcription factors is being referenced in the experiment. \n
        Specifically, you need to find the following:\n
            1). FACTOR: The Official Gene Symbol of the transcription factor (the target protein that ChIP-seq was conducted on) whose binding sites are being mapped on the genome.
        """
    )

    INPUT_PROMPT = (
        """
        Objective: Identify the most likely transcription factor referenced in the provided ChIP-seq metadata.
        The transcription factor being referenced in this metadata must be one of the following official symbols: 
            {}

        Instructions:
        - Analyze the provided metadata and consider only the listed transcription factors.
        - Compare the metadata against known attributes of these transcription factors.
        - From one of the choices provided above identify and output the official symbol that most accurately corresponds to the provided metadata.

        Important:
        - You must output only the Symbol element from your selected item provided list.
        - Do NOT output synonyms, descriptions, or any additional information—only the exact official symbol.
        - The output transcription factor must be one of the provided transcription factors.

        Please choose the factor from this sample:\n
            {}
        \n
        The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
            {}
        \n
        """
    )

    formatted_prompt = INPUT_PROMPT.format(" \n".join(TF_factors),xml_prompt, gse_prompts)
    setup_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=GUIDELINES_PROMPT
            ),
            HumanMessage(
                content=formatted_prompt
            )
        ]
    ) 

    try:
        chat_message = setup_messages.format_messages()
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        res_content = res.content.strip()
        return res_content
    except Exception as e:
        raise
    
def cr_picker(CR_factors, gsm_xml_string, gse_xml_strings, model=None):
    """
    Identifies the most likely chromatin remodeler referenced in ChIP-seq metadata.

    This function analyzes GSM and GSE XML files, simplifies their content, and uses 
    a specified LLM to determine which chromatin remodeler from a given list is most 
    likely referenced in the metadata.

    Args:
        CR_factors (list): A list of chromatin remodeling genes (gene symbols) to consider.
        gsm_xml_string (str): String content of the GSM XML file.
        gse_xml_strings (List[str]): A list of string contents for the GSE XML files.

    Returns:
        str: The gene symbol of the identified chromatin remodeler.
    """


    CR_factors = [str(d) for d in CR_factors]

    # Simplify GSM XML file
    xml_prompt = gsm_xml_string
    gse_prompts = "\n\n".join(gse_xml_strings)

    GUIDELINES_PROMPT = (
        """
        You are an intelligent and accurate Named Entity Recognition (NER) system with a specialization in Genomics and Biology.\n\n

        I will provide you with GSM XML files (referring to individual samples) and GSE XML files (referring to series of GSM samples). \n
        I will also provide you with a list of chromatin remodeling genes that could be referenced in the GSM and GSE XML Files \n\n

        In the XML files provided, your task is to identify which chromatin remodeler from the list of chromatin remodeling genes is being referenced in the experiment. \n
        Specifically, you need to find the following:\n
            1). Chromatin Remodeling Gene: The Official Gene Symbol of the chromatin remodeler (the target protein that ChIP-seq was conducted on) whose binding sites are being mapped on the genome.
        """
    )

    INPUT_PROMPT = (
        """
        Objective: Identify the most likely chromatin remodeler referenced in the provided ChIP-seq metadata.
        The chromatin remodeler being referenced in this metadata must be one of the following official symbols: 
            {}

        Instructions:
        - Analyze the provided metadata and consider only the listed chromatin remodelers.
        - Compare the metadata against known attributes of these chromatin remodelers.
        - From one of the choices provided above identify and output the official symbol that most accurately corresponds to the provided metadata.

        Important:
        - You must output only the Symbol element from your selected item provided list.
        - Do NOT output synonyms, descriptions, or any additional information—only the exact official symbol.
        - The output chromatin remodeler must be one of the provided chromatin remodelers.

        Please choose the chromatin remodeler from this sample:\n
            {}
        \n
        The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
            {}
        \n
        """
    )

    formatted_prompt = INPUT_PROMPT.format(" \n".join(CR_factors),xml_prompt, gse_prompts)
    setup_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=GUIDELINES_PROMPT
            ),
            HumanMessage(
                content=formatted_prompt
            )
        ]
    ) 

    try:
        chat_message = setup_messages.format_messages()
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        res_content = res.content.strip()
        return res_content
    except Exception as e:
        raise
    
def gene_picker(GENE_factors, gsm_xml_string, gse_xml_strings, model=None):
    """
    Identifies the most likely gene referenced in ChIP-seq metadata.

    This function analyzes GSM and GSE XML files, simplifies their content, and uses 
    a specified LLM to determine which gene from a given list is most likely referenced 
    in the metadata.

    Args:
        GENE_factors (list): A list of genes (gene symbols) to consider.
        gsm_xml_string (str): String content of the GSM XML file.
        gse_xml_strings (List[str]): A list of string contents for the GSE XML files.

    Returns:
        str: The gene symbol of the identified gene.

    Raises:
        FileNotFoundError: If the required GSM or GSE XML files are missing.
    """

    GENE_factors = [str(d) for d in GENE_factors]

    # Simplify GSM XML file
    xml_prompt = gsm_xml_string
    gse_prompts = "\n\n".join(gse_xml_strings)

    GUIDELINES_PROMPT = (
        """
        You are an intelligent and accurate Named Entity Recognition (NER) system with a specialization in Genomics and Biology.\n\n

        I will provide you with GSM XML files (referring to individual samples) and GSE XML files (referring to series of GSM samples). \n
        I will also provide you with a list of genes that could be referenced in the GSM and GSE XML Files \n\n

        In the XML files provided, your task is to identify which gene from the list of genes is being referenced in the experiment. \n
        Specifically, you need to find the following:\n
            1). FACTOR: The Official Gene Symbol of the gene (the target protein that ChIP-seq was conducted on) whose binding sites are being mapped on the genome.
        """
    )

    INPUT_PROMPT = (
        """
        Objective: Identify the most likely gene/target protein referenced in the provided ChIP-seq metadata.
        The gene/target protein being referenced in this metadata must be one of the following official symbols: 
            {}

        Instructions:
        - Analyze the provided metadata and consider only the listed gene/target proteins.
        - Compare the metadata against known attributes of these gene/target proteins.
        - From one of the choices provided above identify and output the official symbol that most accurately corresponds to the provided metadata.

        Important:
        - You must output only the Symbol element from your selected item provided list.
        - Do NOT output synonyms, descriptions, or any additional information—only the exact official symbol.
        - The output gene or target protein must be one of the provided gene/target proteins.

        Please choose the gene/target protein from this sample:\n
            {}
        \n
        The sample is one of several in a series of related experiments. The sample belongs to this series' metadata:\n
            {}
        \n
        """
    )

    formatted_prompt = INPUT_PROMPT.format(" \n".join(GENE_factors),xml_prompt, gse_prompts)
    setup_messages = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=GUIDELINES_PROMPT
            ),
            HumanMessage(
                content=formatted_prompt
            )
        ]
    ) 

    try:
        chat_message = setup_messages.format_messages()
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        res_content = res.content.strip()
        return res_content
    except Exception as e:
        raise

def validate_binding_protein(factor, gsm_xml_string, gse_xml_strings, matches, genes_df, TF_df, chromatin_df, model=None):
    """
    Validates and identifies the binding protein (transcription factor, chromatin remodeler, histone modification, or gene) 
    based on the provided factor and GSM metadata. The function first checks for transcription factors, 
    chromatin remodelers, and histone modifications, and then selects the most appropriate match. 
    The process may involve querying an LLM model for further validation.

    Parameters:
    factor (str): The factor (transcription factor, chromatin remodeler, histone modification, or gene) to validate.
    gsm_xml_string (str): String content of the GSM XML file.
    gse_xml_strings (List[str]): A list of string contents for the GSE XML files.
    matches (list): A list of matching entries (genes or factors) from the parsed data.

    Returns:
    tuple: A tuple containing:
        - pd.Series: The validated result containing the factor details (symbol).
        - bool: A boolean indicating whether a valid result was found.
    """
    satisfied = False
    result = pd.Series("none")
    
    try:
        match_count = len(matches)
    except Exception as e:
        print(f"Error counting matches: {e}")
        return result, False

    if match_count == 1:
        try:
            entry = dict(matches[0])
            entry["factor_type"] = classify_human_gene(entry.get("Symbol"), TF_df, chromatin_df)
            result = pd.Series(entry)
            satisfied = True
        except Exception as e:
            print(f"Error processing single match: {e}")
    elif match_count > 1:
        try:
            filtered_matches_tf = validate_transcription_factor(matches, TF_df)
        except Exception as e:
            print(f"Error validating transcription factors: {e}")
            filtered_matches_tf = []

        try:
            if filtered_matches_tf:
                if len(filtered_matches_tf) == 1:
                    entry = dict(filtered_matches_tf[0])
                    entry["factor_type"] = "transcription_factor"
                    result = pd.Series(entry)
                    satisfied = True
                else:
                    selected_factor = tf_picker(filtered_matches_tf, gsm_xml_string, gse_xml_strings, model=model)
                    selected_result = [row for row in filtered_matches_tf if row["Symbol"] == selected_factor]
                    if selected_result:
                        entry = dict(selected_result[0])
                        entry["factor_type"] = "transcription_factor"
                        result = pd.Series(entry)
                        satisfied = True
            else:
                try:
                    filtered_matches_cr = validate_chromatin_remodelers(matches, chromatin_df)
                except Exception as e:
                    print(f"Error validating chromatin remodelers: {e}")
                    filtered_matches_cr = []

                if filtered_matches_cr:
                    if len(filtered_matches_cr) == 1:
                        entry = dict(filtered_matches_cr[0])
                        entry["factor_type"] = "chromatin_remodeler"
                        result = pd.Series(entry)
                        satisfied = True
                    else:
                        try:
                            selected_factor = cr_picker(filtered_matches_cr, gsm_xml_string, gse_xml_strings, model=model)
                            selected_result = [row for row in filtered_matches_cr if row["Symbol"] == selected_factor]
                            if selected_result:
                                entry = dict(selected_result[0])
                                entry["factor_type"] = "chromatin_remodeler"
                                result = pd.Series(entry)
                                satisfied = True
                        except Exception as e:
                            print(f"Error picking chromatin remodeler: {e}")
                else:
                    try:
                        is_HIST = validate_histone_mark(factor)
                        if is_HIST:
                            result = pd.Series({"Symbol": factor, "factor_type": "histone_modification"})
                            satisfied = True
                        else:
                            selected_factor = gene_picker(matches, gsm_xml_string, gse_xml_strings, model=model)
                            selected_result = [row for row in matches if clean_input(row["Symbol"]) == clean_input(selected_factor)]
                            if selected_result:
                                entry = dict(selected_result[0])
                                entry["factor_type"] = classify_human_gene(entry.get("Symbol"), TF_df, chromatin_df)
                                result = pd.Series(entry)
                                satisfied = True
                    except Exception as e:
                        print(f"Error validating histone mark or picking gene: {e}")
        except Exception as e:
            print(f"Error validating: {e}")
    return result, satisfied

def verify_factor(factor, gsm_xml_string, gse_xml_strings, genes_df, TF_df, chromatin_df,
                  viral_index=None, editing_index=None, model=None):
    """
    Verify and classify a candidate factor through a layered cascade.

    Cascade order (first hit wins):
      1. Histone modification rule (``validate_histone_mark``)
      2. Viral factor DB (``validate_viral_factor``)
      3. Gene-editing tool DB (``validate_gene_editing_tool``)
      4. Human gene DB (``match_human_gene`` + ``validate_binding_protein``,
         which now classifies the symbol as TF / CR / gene)
      5. Synonym retry (LLM-generated synonyms, each run through steps 1-4)
      6. ``factor_recheck`` retry loop (max ``max_loops`` iterations)

    Args:
        factor (str): Candidate factor from ``extract_factor``.
        gsm_xml_string (str): GSM MINiML XML (string content).
        gse_xml_strings (List[str]): Parent GSE XML strings.
        genes_df (pd.DataFrame): Human gene info table.
        TF_df (pd.DataFrame): Human TF table.
        chromatin_df: Chromatin remodeler table (DataFrame or dict-like).
        viral_index (dict | None): Lookup index from
            :func:`_build_factor_index` over ``viral_factors.json``.
        editing_index (dict | None): Lookup index over
            ``gene_editing_tools.json``.
        model (str | None): Optional LLM model identifier.

    Returns:
        pd.Series: A Series with at least ``"Symbol"`` and ``"factor_type"``
        keys on success. Empty Series on full cascade failure.
    """
    failed_factors = []
    result = pd.Series(dtype='object')
    max_loops = 2
    loop_count = 0
    satisfied = False
    viral_index = viral_index or {}
    editing_index = editing_index or {}

    def _check_non_gene_dbs(candidate):
        """Run histone -> viral -> gene-editing checks; return Series or None."""
        if not candidate or candidate == "None":
            return None
        try:
            if validate_histone_mark(candidate):
                return pd.Series({"Symbol": candidate, "factor_type": "histone_modification"})
        except Exception as e:
            print(f"Error validating histone mark: {e}")
        try:
            viral_match = validate_viral_factor(candidate, viral_index)
            if viral_match:
                return pd.Series({"Symbol": viral_match, "factor_type": "viral_factor"})
        except Exception as e:
            print(f"Error validating viral factor: {e}")
        try:
            editing_match = validate_gene_editing_tool(candidate, editing_index)
            if editing_match:
                return pd.Series({"Symbol": editing_match, "factor_type": "gene_editing_tool"})
        except Exception as e:
            print(f"Error validating gene-editing tool: {e}")
        return None

    while not satisfied and loop_count < max_loops:
        loop_count += 1
        failed_factors.append(factor)

        # Steps 1-3: rule-based + curated DB checks
        pre_match = _check_non_gene_dbs(factor)
        if pre_match is not None:
            return pre_match

        # Step 4: human gene DB
        try:
            matches = match_human_gene(genes_df, factor) if factor != "None" else []
        except Exception as e:
            print(f"Error matching factor in human gene dataset: {e}")
            matches = []

        try:
            if matches:
                result, satisfied = validate_binding_protein(
                    factor, gsm_xml_string, gse_xml_strings, matches,
                    genes_df, TF_df, chromatin_df, model=model
                )
                if satisfied:
                    return result
        except Exception as e:
            print(f"Error validating binding protein for factor '{factor}': {e}")

        # Step 5: synonyms (each synonym re-runs steps 1-4)
        if not satisfied and factor != "None":
            try:
                synonyms = generate_synonyms(factor, model=model)
            except Exception as e:
                print(f"Error generating/evaluating synonyms for factor '{factor}': {e}")
                synonyms = []

            for synonym in synonyms:
                if satisfied:
                    break
                syn_pre_match = _check_non_gene_dbs(synonym)
                if syn_pre_match is not None:
                    return syn_pre_match
                try:
                    syn_matches = match_human_gene(genes_df, synonym)
                    if syn_matches:
                        result, satisfied = validate_binding_protein(
                            synonym, gsm_xml_string, gse_xml_strings, syn_matches,
                            genes_df, TF_df, chromatin_df, model=model
                        )
                        if satisfied:
                            return result
                except Exception as e:
                    print(f"Error validating synonym '{synonym}': {e}")

        # Step 6: factor_recheck loop — try a different factor name on the next loop pass
        if not satisfied:
            try:
                rechecked_factor, _rechecked_epitope = factor_recheck(
                    gsm_xml_string, gse_xml_strings, failed_factors, model=model
                )
                factor = rechecked_factor
            except Exception as e:
                print(f"Error during factor recheck: {e}")
                break

    return result

def extract_verify_factor(gsm_xml_string, gse_xml_strings, genes_df, TF_df, chromatin_df,
                          viral_index=None, editing_index=None, model=None):
    """
    Extract, verify, and classify the ChIP-seq target for a single GSM.

    Wires the epitope-tag boolean from :func:`extract_factor` and the
    ``factor_type`` from :func:`verify_factor` into a single output dict
    suitable for :func:`_format_output_structure`.

    Args:
        gsm_xml_string (str): Simplified GSM XML body.
        gse_xml_strings (List[str]): Simplified parent GSE XML bodies.
        genes_df (pd.DataFrame): Human gene info table.
        TF_df (pd.DataFrame): Human TF table.
        chromatin_df: Chromatin remodeler table.
        viral_index (dict | None): Viral factor lookup index.
        editing_index (dict | None): Gene-editing tool lookup index.
        model (str | None): Optional LLM model identifier.

    Returns:
        dict: Always contains ``"extracted_factor"`` and ``"factor_type"``.
        Optionally contains ``"factor_status"`` for notable success modifiers
        (``epitope_tagged``) or failure modes (``control_sample``,
        ``no_factor_detected``, ``extraction_failed``, ``verification_failed``,
        ``incomplete_epitope_tag``).
    """
    try:
        control = is_control(gsm_xml_string, model=model)
        if control.strip().lower() == "true":
            return {"extracted_factor": "N/A",
                    "factor_type": "none",
                    "factor_status": "control_sample"}
    except Exception as e:
        print(f"An error occurred detecting control sample: {e}")

    try:
        actual_factor, epitope_tagged = extract_factor(gsm_xml_string, gse_xml_strings, model=model)
    except Exception as e:
        print(f"An error occurred Extracting Factor: {e}")
        return {"extracted_factor": "N/A",
                "factor_type": "none",
                "factor_status": "extraction_failed"}

    if not actual_factor or actual_factor == "None":
        if epitope_tagged:
            return {"extracted_factor": "N/A",
                    "factor_type": "none",
                    "factor_status": "incomplete_epitope_tag"}
        return {"extracted_factor": "N/A",
                "factor_type": "none",
                "factor_status": "no_factor_detected"}

    try:
        verified_factor = verify_factor(
            actual_factor, gsm_xml_string, gse_xml_strings,
            genes_df, TF_df, chromatin_df,
            viral_index=viral_index, editing_index=editing_index, model=model
        )
        if verified_factor is not None and "Symbol" in verified_factor and verified_factor["Symbol"]:
            verified_object = {
                "extracted_factor": verified_factor["Symbol"],
                "factor_type": verified_factor.get("factor_type", "gene"),
            }
            if epitope_tagged:
                verified_object["factor_status"] = "epitope_tagged"
        else:
            verified_object = {"extracted_factor": "N/A",
                               "factor_type": "none",
                               "factor_status": "verification_failed"}
    except Exception as e:
        print(f"An error occured verifying the factor: {e}")
        verified_object = {"extracted_factor": "N/A",
                           "factor_type": "none",
                           "factor_status": "verification_failed"}

    return verified_object

### Fetch-mode pipeline (used when caller omits all three mapping JSONs) ###
def _expand_accessions_to_gsm_queue(accessions, verbose=False):
    """
    Split a mixed list of GSM/GSE accession strings into a deduplicated GSM
    work queue. GSE accessions are fetched and expanded to their child GSMs
    via NCBI; bare GSMs pass through. Order is preserved (first-seen wins).
    """
    queue = []
    seen = set()
    for acc in accessions:
        if not isinstance(acc, str) or not acc:
            continue
        if acc.upper().startswith("GSE"):
            if verbose:
                print(f"Expanding {acc} to child GSMs...", file=sys.stderr)
            children = expand_gse_to_gsms(acc, verbose=verbose)
            if verbose:
                print(f"  {acc} -> {len(children)} child GSM(s)", file=sys.stderr)
            for child in children:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        else:
            if acc not in seen:
                seen.add(acc)
                queue.append(acc)
    return queue


def _fetch_gsm_and_gse_text(gsm_id, verbose=False):
    """
    Fetch a single GSM's MINiML XML and the XMLs of every parent GSE we can
    discover, then run them through the existing simplifiers.

    Returns:
        tuple[str, str] | tuple[None, None]:
            (gsm_text, gse_text) on success — gse_text may be "" if no parent
            GSE could be fetched. (None, None) when the GSM itself could not
            be fetched after MAX_TRIES (caller should emit a blank entry).
    """
    gsm_xml, parent_gses = discover_parent_gses(gsm_id, verbose=verbose)
    if gsm_xml is None:
        return None, None

    try:
        gsm_text = simplify_gsm_xml_file(gsm_xml)
    except Exception as e:
        print(f"{gsm_id} | Error simplifying fetched GSM XML: {e}", file=sys.stderr)
        return None, None

    gse_prompts = []
    for gse_id in parent_gses:
        gse_xml = fetch_geo_xml(gse_id, targ="self", verbose=verbose)
        if gse_xml is None:
            # final-failure already printed by fetch_geo_xml; carry on
            continue
        try:
            gse_prompts.append(simplify_gse_xml_file(gse_xml))
        except Exception as e:
            print(f"{gsm_id} | Error simplifying fetched GSE XML for {gse_id}: {e}", file=sys.stderr)
            continue

    return gsm_text, "\n\n".join(gse_prompts)


### Factor Extraction Functionality ###
def meta_extract_factors(gsm_ids_input, gsm_to_gse_path=None, gsm_paths_path=None, gse_paths_path=None, model=None, verbose=False):
    """
    Extract verified factors from GSM (and optionally GSE) accessions.

    Two modes:

    1. **Fetch mode** (recommended) — omit all three mapping path arguments.
       Bare accession strings in ``gsm_ids_input`` are fetched directly from
       NCBI GEO. GSE accessions are auto-expanded to their child GSMs.
       After ``MAX_TRIES`` retries, an unrecoverable accession is reported
       on stderr and emitted as ``{gsm_id: {}}`` so output positions are
       preserved.

    2. **Local-file mode** — provide all three mapping JSONs. Same behavior
       as before this change.

    Args:
        gsm_ids_input: List of accession strings (GSM/GSE), or a path to a
            JSON file containing such a list.
        gsm_to_gse_path (str, optional): Path to JSON mapping GSM IDs to GSE IDs.
        gsm_paths_path (str, optional): Path to JSON mapping GSM IDs to file paths.
        gse_paths_path (str, optional): Path to JSON mapping GSE IDs to file paths.
        model (str, optional): LLM identifier in ``provider:model`` form.
        verbose (bool): If True, emit retry/expansion progress to stderr.

    Returns:
        list: A list of dicts with the standardized output format.
    """
    fetch_mode = (gsm_to_gse_path is None and gsm_paths_path is None and gse_paths_path is None)
    if not fetch_mode and not all([gsm_to_gse_path, gsm_paths_path, gse_paths_path]):
        raise ValueError(
            "Provide all three mapping paths (gsm_to_gse, gsm_paths, gse_paths) "
            "to use local-file mode, or omit all three to fetch from NCBI GEO."
        )

    # Parse GSM IDs input
    gsm_ids = _parse_gsm_ids_input(gsm_ids_input)
    if not gsm_ids:
        print("No valid GSM IDs provided.")
        return []

    # Load validation data once
    data_dir = get_data_dir()
    parsed_factor_dir = data_dir / "parsed_factor_data"

    try:
        genes_df = pd.read_csv(parsed_factor_dir / "gene_info.csv", encoding='utf-8')
        TF_df = pd.read_csv(parsed_factor_dir / "Homo_sapiens_TF.csv", sep="\t", encoding='utf-8')
        chromatin_df = pd.read_csv(parsed_factor_dir / "Homo_sapiens_CR.csv", encoding='utf-8')
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return []
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV files: {e}")
        return []

    try:
        with open(parsed_factor_dir / "viral_factors.json", "r", encoding='utf-8') as f:
            viral_db = json.load(f)
        with open(parsed_factor_dir / "gene_editing_tools.json", "r", encoding='utf-8') as f:
            editing_db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading non-human factor DBs: {e}")
        viral_db, editing_db = {}, {}

    viral_index = _build_factor_index(viral_db)
    editing_index = _build_factor_index(editing_db)

    results = []

    if fetch_mode:
        gsm_queue = _expand_accessions_to_gsm_queue(gsm_ids, verbose=verbose)
        for gsm_id in gsm_queue:
            gsm_text, gse_text = _fetch_gsm_and_gse_text(gsm_id, verbose=verbose)
            if gsm_text is None:
                # fetch_geo_xml already logged the final failure
                results.append({gsm_id: {}})
                continue
            try:
                factor_result = extract_verify_factor(
                    gsm_text, gse_text, genes_df, TF_df, chromatin_df,
                    viral_index=viral_index, editing_index=editing_index, model=model
                )
                if factor_result and "extracted_factor" in factor_result:
                    formatted_result = _format_output_structure(
                        gsm_id,
                        extracted_factor=factor_result["extracted_factor"],
                        factor_status=factor_result.get("factor_status"),
                        factor_type=factor_result.get("factor_type")
                    )
                    results.append(formatted_result)
                if verbose:
                    print(f"Extracted factor for {gsm_id}", file=sys.stderr)
            except Exception as e:
                print(f"Error extracting target protein from {gsm_id}: {e}", file=sys.stderr)
                continue
        return results

    # Local-file mode (mappings provided)
    gsm_to_gse = _load_json_data(gsm_to_gse_path)
    gsm_paths = _load_json_data(gsm_paths_path)
    gse_paths = _load_json_data(gse_paths_path)

    if not all([gsm_to_gse, gsm_paths, gse_paths]):
        print("Failed to load one or more lookup mapping files.")
        return []

    for gsm_id in gsm_ids:
        # Get GSM file path
        gsm_file_path = gsm_paths.get(gsm_id)
        if not gsm_file_path or not os.path.exists(gsm_file_path):
            print(f"GSM file not found for {gsm_id}")
            continue

        # Get associated GSE IDs and their file paths
        gse_ids = gsm_to_gse.get(gsm_id, [])

        # Collect GSE content
        if gse_ids:
            gse_prompts = []
            for gse_id in gse_ids:
                gse_file_path = gse_paths.get(gse_id)
                if not gse_file_path or not os.path.exists(gse_file_path):
                    print(f"GSE XML file not found for {gse_id}")
                    continue
                try:
                    gse_prompts.append(simplify_gse_xml_file(gse_file_path))
                except Exception as e:
                    print(f"Error simplifying GSE XML file '{gse_file_path}': {e}")
                    continue

            gse_text = "\n\n".join(gse_prompts)
        else:
            gse_text = ""

        # Simplify GSM XML file
        try:
            gsm_file = simplify_gsm_xml_file(gsm_file_path)
        except Exception as e:
            print(f"{gsm_id} | Error simplifying GSM XML file '{gsm_file_path}': {e}")
            continue

        # Extract factor
        try:
            factor_result = extract_verify_factor(
                gsm_file, gse_text, genes_df, TF_df, chromatin_df,
                viral_index=viral_index, editing_index=editing_index, model=model
            )
            if factor_result and "extracted_factor" in factor_result:
                formatted_result = _format_output_structure(
                    gsm_id,
                    extracted_factor=factor_result["extracted_factor"],
                    factor_status=factor_result.get("factor_status"),
                    factor_type=factor_result.get("factor_type")
                )
                results.append(formatted_result)
            print(f"Extracted factor for {gsm_id}")
        except Exception as e:
            print(f"Error extracting target protein from {gsm_id}: {e}")
            continue

    return results

### Extract Ontologies ###
def extract_structured_ontology(gsm_xml_string, gse_xml_strings, model=None):
    """
    Extract cell line, cell type, tissue, and disease ontologies for a ChIP-seq sample from GSM (sample-level)
    and GSE (series-level) XML metadata using an LLM with a strict structured output schema.

    Output fields follow these conventions:
      - cell_line: Cellosaurus official cell line symbol (e.g., "MCF7", "T47D")
      - cell_type: standardized cell type term aligned to EFO/Uberon when applicable
      - tissue: standardized anatomical source term aligned to Uberon/EFO when applicable
      - disease: standardized disease term when present

    Rules: use official ontology naming (no parenthetical synonyms); if a field is not supported by the metadata,
    return "N/A" for that field.

    Args:
        gsm_xml_string: GSM XML string for a single sample.
        gse_xml_strings: List of GSE XML strings providing experiment/series context.

    Returns:
        ChIPSeqMetadata (Pydantic model): {cell_line, cell_type, tissue, disease}.
    """

    class ChIPSeqMetadata(BaseModel):
        cell_line: str = Field(description="The official cell line symbol name, that corresponds to the Cellosaurus official Symbol of the cell line.")
        cell_type: str = Field(description="The specific type of cell used in the experiment, validated against the Experimental Factor Ontology (EFO) and Uberon. This should describe the biological characteristics and classification of the cell, such as 'T-cell', 'fibroblast', or 'neuron'.")
        tissue: str = Field(description="The anatomical source or origin of the cells, validated against the Experimental Factor Ontology (EFO) and Uberon Ontology. This should describe the tissue type, such as 'lung', 'brain', or 'liver', from which the cells were derived.")
        disease: str = Field(description="The disease or pathological condition associated with the cell line or tissue, validated against the Experimental Factor Ontology (EFO) and Uberon Ontology. This should describe the disease context, such as 'breast cancer', 'Alzheimer's disease', or 'type 2 diabetes'.")
        
    # Simplify GSM XML file
    xml_prompt = gsm_xml_string
    gse_prompts = "\n\n".join(gse_xml_strings)

    # Ontology guidelines are byte-identical across calls so caching kicks in
    # on every supported provider. Note: structured_output returns the parsed
    # Pydantic model directly, so we don't attempt _log_cache_usage on the
    # result (no usage_metadata is attached) — the factor calls already prove
    # the cache is firing when CISTROMEMX_CACHE_DEBUG is set.
    formatted_input = _ONTOLOGY_INPUT_TEMPLATE.format(xml_prompt, gse_prompts)
    chat_message = [
        _make_cached_system_message(_ONTOLOGY_GUIDELINES, model),
        HumanMessage(content=formatted_input),
    ]

    try:
        llm = _init_llm(model)
        # OpenAI's default "json_schema" structured output path can attach the
        # parsed Pydantic object to the raw SDK response and trigger noisy
        # Pydantic serializer warnings. Tool/function calling returns the same
        # validated model without that raw parsed-response serialization.
        structured_llm = llm.with_structured_output(
            ChIPSeqMetadata,
            method="function_calling",
        )
        res = structured_llm.invoke(chat_message)
        return res
    except Exception as e:
        raise

def validate_ontology(extracted_obj, cellosaurus_index, efo_index, uberon_index, remove_words=False, validated=None):
    """Validates ontology terms and returns a new object with all matching ontology metadata."""
    if validated is None:
        validated = {}
    def ensure_list(value):
        if isinstance(value, list):
            return value
        return [value] if isinstance(value, dict) else []

    for key in ["cell_line", "cell_type", "tissue", "disease"]:
        if extracted_obj.get(key) == "N/A" or validated.get(key) is not None:
            validated.setdefault(key, None)
            continue

        term_key = process_string(extracted_obj.get(key), remove=remove_words)
        if not term_key:
            validated[key] = None
            continue

        matches = []
        if term_key in cellosaurus_index:
            for match in ensure_list(cellosaurus_index[term_key]):
                matches.append({
                    **match,
                    "term": extracted_obj[key],
                    "term_identity": key
                })
        if term_key in efo_index:
            for match in ensure_list(efo_index[term_key]):
                matches.append({
                    **match,
                    "term": extracted_obj[key],
                    "term_identity": key
                })
        if term_key in uberon_index:
            for match in ensure_list(uberon_index[term_key]):
                matches.append({
                    **match,
                    "term": extracted_obj[key],
                    "term_identity": key
                })

        validated[key] = matches if matches else None

    return validated

def validate_ontology_fuzzy(extracted_obj, cellosaurus_index, efo_index, uberon_index, validated=None):
    """Validates ontology terms using fuzzy matching and returns structured ontology metadata."""
    if validated is None:
        validated = {}

    def ensure_list(value):
        if isinstance(value, list):
            return value
        return [value] if isinstance(value, dict) else []

    def fuzzy_match(term, index):
        matches = []
        max_score = 0.85
        for key, values in index.items():
            score = fuzz.token_sort_ratio(term, key) / 100
            if score >= max_score:
                if score > max_score:
                    matches = values
                    max_score = score
                elif score == max_score:
                    matches.extend(values)
        return matches if matches else None

    for key in ["cell_line", "cell_type", "tissue", "disease"]:
        if extracted_obj.get(key) == "N/A" or validated.get(key) is not None:
            validated.setdefault(key, None)
            continue

        term_key = clean_input_fuzzy(extracted_obj.get(key))
        if not term_key:
            validated[key] = None
            continue

        matches = []
        cellosaurus_matches = fuzzy_match(term_key, cellosaurus_index)
        if cellosaurus_matches:
            for match in ensure_list(cellosaurus_matches):
                matches.append({
                    **match,
                    "term": extracted_obj[key],
                    "term_identity": key
                })
        efo_matches = fuzzy_match(term_key, efo_index)
        if efo_matches:
            for match in ensure_list(efo_matches):
                matches.append({
                    **match,
                    "term": extracted_obj[key],
                    "term_identity": key
                })
        uberon_matches = fuzzy_match(term_key, uberon_index)
        if uberon_matches:
            for match in ensure_list(uberon_matches):
                matches.append({
                    **match,
                    "term": extracted_obj[key],
                    "term_identity": key
                })

        validated[key] = matches if matches else None
    return validated

def process_ontology(
    extracted_object,
    cellosaurus_index, efo_index, uberon_index,
    cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
    cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index
):
    """
    Process ontology with strict, reduced, and fuzzy matching in staged order:
    1. Full index without word removal.
    2. Full index with word removal (search only).
    3. Reduced index without word removal.
    4. Reduced index with word removal (search only).
    5. Fuzzy match on remaining unmatched non-N/A fields.
    """
    ont_object = extracted_object.get("ontologies", {}).get("extracted_ontologies", {})
    validated_object = {}

    # Step 1: Try full index, no word removal
    validated_object = validate_ontology(
        ont_object, cellosaurus_index, efo_index, uberon_index,
        remove_words=False, validated=validated_object
    )

    # Step 2: Try full index, with word removal (search only)
    validated_object = validate_ontology(
        ont_object, cellosaurus_index, efo_index, uberon_index,
        remove_words=True, validated=validated_object
    )

    # Step 3: Try reduced index, no word removal
    validated_object = validate_ontology(
        ont_object, cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
        remove_words=False, validated=validated_object
    )

    # Step 4: Try reduced index, with word removal
    validated_object = validate_ontology(
        ont_object, cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
        remove_words=True, validated=validated_object
    )

    # Step 5: Fuzzy match only on fields still unmatched and not N/A
    validated_object = validate_ontology_fuzzy(
        ont_object, cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index,
        validated=validated_object
    )

    validated_object["geo"] = extracted_object.get("geo", "N/A")
    validated_object["extracted"] = ont_object

    return validated_object

def generate_alternate_names(ontology, model=None):
    """
    Generates alternative names for a cell ontology term that did not return an exact match from the databases.

    This function takes an input ontology term and attempts to generate three alternative names by applying various rules such as simplifying terms, swapping synonyms, removing unnecessary words (e.g., "Human" or "cell"), and ensuring the names align with standard naming conventions. The output is a list of alternative names formatted for better matching with databases such as Cellosaurus, EFO, or Uberon.

    Args:
        ontology (str): The original cell ontology term that did not match.

    Returns:
        str: A string containing three alternative names for the input cell ontology, formatted as a list.
    """

    INPUT_PROMPT = (
    """
    You are an intelligent and accurate system with a specialization in Genomics and Biology.\n\n

    Your job is to receive an input cell ontology that did not return an exact match from the Cellosaurus, Experimental Factor Ontology (EFO), or Uberon databases. \n
    Using your knowledge of these databases and their naming conventions, suggest alternative names for the input ontology to improve matching. \n\n

    Follow these rules to generate the alternatives: \n\n

    Rules: \n
    1. **Singularity/Plurality**: If the input ontology is plural (e.g., "mammary epithelial cells"), suggest a singular form (e.g., "mammary epithelial cell").\n
    2. **Synonyms**: Replace words with common synonyms that align with biological terminology. For instance:\n
        - "Human fetal kidney" → "Embryonic kidney"\n
        - "Epithelial-like" → "Epithelial"\n
    3. **Simplification**: Before applying synonyms, simplify the ontology name by removing unnecessary descriptors:\n
        - "Human B-lymphocyte cell line" → "B-lymphocyte cell line" or "B-lymphocyte"\n
    4. **Descriptor Removal**: Remove general terms like "Human," "cell," or "cell line" where applicable:\n
        - "Invasive lobular carcinoma cells" → "Invasive lobular carcinoma"\n
    5. **Balanced Specificity**: Avoid overly broad generalizations or unnecessary specificity:\n
        - Incorrect: "Breast cancer" → "Invasive ductal carcinoma of the breast"\n
        - Correct: "Breast cancer" → "Adenocarcinoma" or "Breast carcinoma"\n
    6. **Parentheses Handling**: If the name includes parentheses, split the content into separate suggestions:\n
        - "Breast cancer (adenocarcinoma)" → ["Breast cancer", "Adenocarcinoma"]\n
    7. **Authenticity**: Maintain the authenticity of the original name. Do not drastically alter its meaning or context.\n\n

    Your task is to generate **three plausible alternative names** for the input ontology that could match the naming conventions of Cellosaurus, EFO, or Uberon. Ensure that your output adheres to the rules above.\n\n

    Output Format:\n
    ["alternative1", "alternative2", "alternative3"]\n\n

    Now find alternative names for this cell ontology: {}\n
    """
    )
    formatted_prompt = INPUT_PROMPT.format(ontology)
    setup_messages = ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                content=formatted_prompt
            )
        ]
    ) 


    try:
        chat_message = setup_messages.format_messages()
        llm = _init_llm(model)
        res = llm.invoke(chat_message)
        actual_array = _parse_llm_list(res.content)
        return actual_array
    except Exception as e:
        raise

def verify_ontology(
    input_ontology,
    cellosaurus_index, efo_index, uberon_index,
    cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
    cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index,
    model=None
):
    """
    Verifies and resolves missing ontology terms using process_ontology and generate_alternate_names.
    (Implementation remains the same as original)
    """
    def is_incomplete(validated):
        return any(value is None for key, value in validated.items() if key not in ("geo", "extracted"))
    
    # Initial attempt
    validated = process_ontology(
        input_ontology,
        cellosaurus_index, efo_index, uberon_index,
        cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
        cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index
    )
    if not is_incomplete(validated):
        return validated

    completed_output = validated
    extracted = validated.get("extracted", {})
    geo = validated.get("geo")

    for key in ["cell_line", "cell_type", "tissue", "disease"]:
        extracted_value = extracted.get(key)
        verified_value = validated.get(key)

        if extracted_value and extracted_value != "N/A" and verified_value is None:
            alt_names = generate_alternate_names(extracted_value, model=model)
            for alt_name in alt_names:
                new_input = {
                    "ontologies": {
                        "extracted_ontologies": {
                            "cell_line": "N/A",
                            "cell_type": "N/A",
                            "tissue": "N/A",
                            "disease": "N/A"
                        }
                    },
                    "geo": geo
                }
                new_input["ontologies"]["extracted_ontologies"][key] = alt_name

                result = process_ontology(
                    new_input,
                    cellosaurus_index, efo_index, uberon_index,
                    cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
                    cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index
                )

                if result.get(key):
                    completed_output[key] = result[key]
                    break

    for key in ["cell_line", "cell_type", "tissue", "disease"]:
        value = completed_output.get(key)
        if isinstance(value, list):
            completed_output[key] = collapse_ontology_terms(value)
             
    return completed_output

def extract_verify_ontology(gsm_id, gsm_xml_string, gse_xml_strings,
                            cellosaurus_index, efo_index, uberon_index,
                            cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
                            cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index,
                            model=None):
    """
    Extracts and verifies ontology terms from GSM and GSE XML strings.
    Updated to use GSM ID instead of file path.
    """
    try:
        structured_object = extract_structured_ontology(gsm_xml_string, gse_xml_strings, model=model)
        extracted_object = {
            "cell_line": structured_object.cell_line,
            "cell_type": structured_object.cell_type,
            "tissue": structured_object.tissue,
            "disease": structured_object.disease
        }
    except Exception as e:
        print(f"An error occurred Extracting Ontology: {e}")    
        return None

    ret_object = {
        "ontologies": {
            "extracted_ontologies": extracted_object
        },
        "geo": gsm_id
    }
    
    try:
        validated_object = verify_ontology( 
            ret_object,
            cellosaurus_index, efo_index, uberon_index,
            cellosaurus_reduce_index, efo_reduce_index, uberon_reduce_index,
            cellosaurus_fuzzy_index, efo_fuzzy_index, uberon_fuzzy_index,
            model=model
        )
    except Exception as e:
        print(f"An error occurred verifying the ontologies: {e}")
        return None
   
    return validated_object

### Ontology Extraction Functionality###
def meta_extract_ontologies(gsm_ids_input, gsm_to_gse_path=None, gsm_paths_path=None, gse_paths_path=None, model=None, verbose=False):
    """
    Extract ontology metadata from GSM (and optionally GSE) accessions.

    Two modes — see :func:`meta_extract_factors` for full details:

    * **Fetch mode** (recommended): omit all three mapping paths; bare
      accession strings are fetched directly from NCBI GEO. GSE accessions
      auto-expand to child GSMs. Unrecoverable accessions are emitted as
      ``{gsm_id: {}}``.
    * **Local-file mode**: provide all three mapping JSONs.

    Args:
        gsm_ids_input: List of accession strings, or path to a JSON list.
        gsm_to_gse_path (str, optional): GSM→GSE mapping JSON.
        gsm_paths_path (str, optional): GSM→XML path mapping JSON.
        gse_paths_path (str, optional): GSE→XML path mapping JSON.
        model (str, optional): LLM identifier in ``provider:model`` form.
        verbose (bool): If True, emit retry/expansion progress to stderr.

    Returns:
        list: A list of ontology extraction dicts with standardized format.
    """
    fetch_mode = (gsm_to_gse_path is None and gsm_paths_path is None and gse_paths_path is None)
    if not fetch_mode and not all([gsm_to_gse_path, gsm_paths_path, gse_paths_path]):
        raise ValueError(
            "Provide all three mapping paths (gsm_to_gse, gsm_paths, gse_paths) "
            "to use local-file mode, or omit all three to fetch from NCBI GEO."
        )

    # Parse GSM IDs input
    gsm_ids = _parse_gsm_ids_input(gsm_ids_input)
    if not gsm_ids:
        print("No valid GSM IDs provided.")
        return []

    # Load ontology data once
    data_dir = get_data_dir()
    parsed_ontology_dir = data_dir / "parsed_ontology_data"

    try:
        with open(parsed_ontology_dir / "cellosaurus.json", "r", encoding='utf-8') as file:
            cellosaurus = json.load(file)
        with open(parsed_ontology_dir / "cellosaurus_reduce.json", "r", encoding='utf-8') as file:
            cellosaurus_reduce = json.load(file)
        with open(parsed_ontology_dir / "cellosaurus_fuzzy.json", "r", encoding='utf-8') as file:
            cellosaurus_fuzzy = json.load(file)

        with open(parsed_ontology_dir / "efo.json", "r", encoding='utf-8') as file:
            efo = json.load(file)
        with open(parsed_ontology_dir / "efo_reduce.json", "r", encoding='utf-8') as file:
            efo_reduce = json.load(file)
        with open(parsed_ontology_dir / "efo_fuzzy.json", "r", encoding='utf-8') as file:
            efo_fuzzy = json.load(file)

        with open(parsed_ontology_dir / "uberon.json", "r", encoding='utf-8') as file:
            uberon = json.load(file)
        with open(parsed_ontology_dir / "uberon_reduce.json", "r", encoding='utf-8') as file:
            uberon_reduce = json.load(file)
        with open(parsed_ontology_dir / "uberon_fuzzy.json", "r", encoding='utf-8') as file:
            uberon_fuzzy = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading ontology files: {e}")
        return []

    results = []

    if fetch_mode:
        gsm_queue = _expand_accessions_to_gsm_queue(gsm_ids, verbose=verbose)
        for gsm_id in gsm_queue:
            gsm_text, gse_text = _fetch_gsm_and_gse_text(gsm_id, verbose=verbose)
            if gsm_text is None:
                results.append({gsm_id: {}})
                continue
            try:
                result = extract_verify_ontology(
                    gsm_id, gsm_text, gse_text,
                    cellosaurus, efo, uberon,
                    cellosaurus_reduce, efo_reduce, uberon_reduce,
                    cellosaurus_fuzzy, efo_fuzzy, uberon_fuzzy,
                    model=model
                )
                if result:
                    formatted_result = _format_output_structure(
                        gsm_id,
                        extracted_ontology=result
                    )
                    results.append(formatted_result)
                if verbose:
                    print(f"Extracted ontologies for {gsm_id}", file=sys.stderr)
            except Exception as e:
                print(f"An error occurred verifying ontologies for {gsm_id}: {e}", file=sys.stderr)
                continue
        return results

    # Local-file mode
    gsm_to_gse = _load_json_data(gsm_to_gse_path)
    gsm_paths = _load_json_data(gsm_paths_path)
    gse_paths = _load_json_data(gse_paths_path)

    if not all([gsm_to_gse, gsm_paths, gse_paths]):
        print("Failed to load one or more lookup mapping files.")
        return []

    for gsm_id in gsm_ids:
        # Get GSM file path
        gsm_file_path = gsm_paths.get(gsm_id)
        if not gsm_file_path or not os.path.exists(gsm_file_path):
            print(f"GSM file not found for {gsm_id}")
            continue

        # Get associated GSE IDs and their file paths
        gse_ids = gsm_to_gse.get(gsm_id, [])

        # Collect GSE content
        if gse_ids:
            gse_prompts = []
            for gse_id in gse_ids:
                gse_file_path = gse_paths.get(gse_id)
                if not gse_file_path or not os.path.exists(gse_file_path):
                    print(f"GSE XML file not found for {gse_id}")
                    continue
                try:
                    gse_prompts.append(simplify_gse_xml_file(gse_file_path))
                except Exception as e:
                    print(f"{gsm_id} | Error simplifying GSE XML file '{gse_file_path}': {e}")
                    continue

            gse_text = "\n\n".join(gse_prompts)
        else:
            gse_text = ""

        # Simplify GSM
        try:
            gsm_text = simplify_gsm_xml_file(gsm_file_path)
        except Exception as e:
            print(f"{gsm_id} | Error simplifying GSM XML file '{gsm_file_path}': {e}")
            continue

        # Extract ontology
        try:
            result = extract_verify_ontology(
                gsm_id, gsm_text, gse_text,
                cellosaurus, efo, uberon,
                cellosaurus_reduce, efo_reduce, uberon_reduce,
                cellosaurus_fuzzy, efo_fuzzy, uberon_fuzzy,
                model=model
            )
            if result:
                formatted_result = _format_output_structure(
                    gsm_id,
                    extracted_ontology=result
                )
                results.append(formatted_result)
            print(f"Extracted ontologies for {gsm_id}")
        except Exception as e:
            print(f"An error occurred verifying ontologies for {gsm_id}: {e}")
            continue

    return results

### Combined Extraction and Verification ###
def meta_extract_factors_and_ontologies(gsm_ids_input, gsm_to_gse_path=None, gsm_paths_path=None, gse_paths_path=None, model=None, verbose=False):
    """
    Extract both factors and ontology metadata from GSM (and optionally GSE) accessions.

    Two modes — see :func:`meta_extract_factors` for full details:

    * **Fetch mode** (recommended): omit all three mapping paths; bare
      accession strings are fetched directly from NCBI GEO. GSE accessions
      auto-expand to child GSMs. Unrecoverable accessions are emitted as
      ``{gsm_id: {}}``.
    * **Local-file mode**: provide all three mapping JSONs.

    Args:
        gsm_ids_input: List of accession strings, or path to a JSON list.
        gsm_to_gse_path (str, optional): GSM→GSE mapping JSON.
        gsm_paths_path (str, optional): GSM→XML path mapping JSON.
        gse_paths_path (str, optional): GSE→XML path mapping JSON.
        model (str, optional): LLM identifier in ``provider:model`` form.
        verbose (bool): If True, emit retry/expansion progress to stderr.

    Returns:
        list of dicts with standardized output format.
    """
    fetch_mode = (gsm_to_gse_path is None and gsm_paths_path is None and gse_paths_path is None)
    if not fetch_mode and not all([gsm_to_gse_path, gsm_paths_path, gse_paths_path]):
        raise ValueError(
            "Provide all three mapping paths (gsm_to_gse, gsm_paths, gse_paths) "
            "to use local-file mode, or omit all three to fetch from NCBI GEO."
        )

    # Parse GSM IDs input
    gsm_ids = _parse_gsm_ids_input(gsm_ids_input)
    if not gsm_ids:
        print("No valid GSM IDs provided.")
        return []

    # Load factor + ontology validation data
    data_dir = get_data_dir()
    parsed_factor_dir = data_dir / "parsed_factor_data"
    parsed_ontology_dir = data_dir / "parsed_ontology_data"

    try:
        genes_df = pd.read_csv(parsed_factor_dir / "gene_info.csv", encoding='utf-8')
        TF_df = pd.read_csv(parsed_factor_dir / "Homo_sapiens_TF.csv", sep="\t", encoding='utf-8')
        chromatin_df = pd.read_csv(parsed_factor_dir / "Homo_sapiens_CR.csv", encoding='utf-8')

        with open(parsed_ontology_dir / "cellosaurus.json", "r", encoding='utf-8') as file:
            cellosaurus = json.load(file)
        with open(parsed_ontology_dir / "cellosaurus_reduce.json", "r", encoding='utf-8') as file:
            cellosaurus_reduce = json.load(file)
        with open(parsed_ontology_dir / "cellosaurus_fuzzy.json", "r", encoding='utf-8') as file:
            cellosaurus_fuzzy = json.load(file)

        with open(parsed_ontology_dir / "efo.json", "r", encoding='utf-8') as file:
            efo = json.load(file)
        with open(parsed_ontology_dir / "efo_reduce.json", "r", encoding='utf-8') as file:
            efo_reduce = json.load(file)
        with open(parsed_ontology_dir / "efo_fuzzy.json", "r", encoding='utf-8') as file:
            efo_fuzzy = json.load(file)

        with open(parsed_ontology_dir / "uberon.json", "r", encoding='utf-8') as file:
            uberon = json.load(file)
        with open(parsed_ontology_dir / "uberon_reduce.json", "r", encoding='utf-8') as file:
            uberon_reduce = json.load(file)
        with open(parsed_ontology_dir / "uberon_fuzzy.json", "r", encoding='utf-8') as file:
            uberon_fuzzy = json.load(file)

    except (FileNotFoundError, pd.errors.ParserError, json.JSONDecodeError) as e:
        print(f"Error loading validation data: {e}")
        return []

    try:
        with open(parsed_factor_dir / "viral_factors.json", "r", encoding='utf-8') as f:
            viral_db = json.load(f)
        with open(parsed_factor_dir / "gene_editing_tools.json", "r", encoding='utf-8') as f:
            editing_db = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading non-human factor DBs: {e}")
        viral_db, editing_db = {}, {}

    viral_index = _build_factor_index(viral_db)
    editing_index = _build_factor_index(editing_db)

    results = []

    if fetch_mode:
        gsm_queue = _expand_accessions_to_gsm_queue(gsm_ids, verbose=verbose)
        for gsm_id in gsm_queue:
            gsm_text, gse_text = _fetch_gsm_and_gse_text(gsm_id, verbose=verbose)
            if gsm_text is None:
                results.append({gsm_id: {}})
                continue

            extracted_factor = None
            factor_status = None
            factor_type = None
            try:
                factor_result = extract_verify_factor(
                    gsm_text, gse_text, genes_df, TF_df, chromatin_df,
                    viral_index=viral_index, editing_index=editing_index, model=model
                )
                if factor_result and "extracted_factor" in factor_result:
                    extracted_factor = factor_result["extracted_factor"]
                    factor_status = factor_result.get("factor_status")
                    factor_type = factor_result.get("factor_type")
            except Exception as e:
                print(f"Error extracting factor from {gsm_id}: {e}", file=sys.stderr)

            extracted_ontology = None
            try:
                extracted_ontology = extract_verify_ontology(
                    gsm_id, gsm_text, gse_text,
                    cellosaurus, efo, uberon,
                    cellosaurus_reduce, efo_reduce, uberon_reduce,
                    cellosaurus_fuzzy, efo_fuzzy, uberon_fuzzy,
                    model=model
                )
            except Exception as e:
                print(f"Error extracting ontology from {gsm_id}: {e}", file=sys.stderr)

            if extracted_factor or extracted_ontology:
                formatted_result = _format_output_structure(
                    gsm_id,
                    extracted_factor=extracted_factor,
                    factor_status=factor_status,
                    factor_type=factor_type,
                    extracted_ontology=extracted_ontology
                )
                results.append(formatted_result)
                if verbose:
                    print(f"Extracted {gsm_id}", file=sys.stderr)
        return results

    # Local-file mode
    gsm_to_gse = _load_json_data(gsm_to_gse_path)
    gsm_paths = _load_json_data(gsm_paths_path)
    gse_paths = _load_json_data(gse_paths_path)

    if not all([gsm_to_gse, gsm_paths, gse_paths]):
        print("Failed to load one or more lookup mapping files.")
        return []

    for gsm_id in gsm_ids:
        # Get GSM file path
        gsm_file_path = gsm_paths.get(gsm_id)
        if not gsm_file_path or not os.path.exists(gsm_file_path):
            print(f"GSM file not found for {gsm_id}")
            continue



        # Load and simplify GSM
        try:
            gsm_text = simplify_gsm_xml_file(gsm_file_path)
        except Exception as e:
            print(f"{gsm_id} | Error simplifying GSM XML file '{gsm_file_path}': {e}")
            continue

        # Get associated GSE IDs and their file paths
        gse_ids = gsm_to_gse.get(gsm_id, [])

        # Collect GSE content
        if gse_ids:
            gse_prompts = []
            for gse_id in gse_ids:
                gse_file_path = gse_paths.get(gse_id)
                if not gse_file_path or not os.path.exists(gse_file_path):
                    print(f"GSE XML file not found for {gse_id}")
                    continue
                try:
                    gse_prompts.append(simplify_gse_xml_file(gse_file_path))
                except Exception as e:
                    print(f"{gsm_id} | Error simplifying GSE XML file '{gse_file_path}': {e}")
                    continue

            gse_text = "\n\n".join(gse_prompts)
        else:
            gse_text = ""

        # Extract factor
        extracted_factor = None
        factor_status = None
        factor_type = None
        try:
            factor_result = extract_verify_factor(
                gsm_text, gse_text, genes_df, TF_df, chromatin_df,
                viral_index=viral_index, editing_index=editing_index, model=model
            )
            if factor_result and "extracted_factor" in factor_result:
                extracted_factor = factor_result["extracted_factor"]
                factor_status = factor_result.get("factor_status")
                factor_type = factor_result.get("factor_type")
        except Exception as e:
            print(f"Error extracting factor from {gsm_id}: {e}")

        # Extract ontology
        extracted_ontology = None
        try:
            extracted_ontology = extract_verify_ontology(
                gsm_id, gsm_text, gse_text,
                cellosaurus, efo, uberon,
                cellosaurus_reduce, efo_reduce, uberon_reduce,
                cellosaurus_fuzzy, efo_fuzzy, uberon_fuzzy,
                model=model
            )
        except Exception as e:
            print(f"Error extracting ontology from {gsm_id}: {e}")

        # Combine if either factor or ontology is present
        if extracted_factor or extracted_ontology:
            formatted_result = _format_output_structure(
                gsm_id,
                extracted_factor=extracted_factor,
                factor_status=factor_status,
                factor_type=factor_type,
                extracted_ontology=extracted_ontology
            )
            results.append(formatted_result)
            print(f"Extracted {gsm_id}")

    return results
