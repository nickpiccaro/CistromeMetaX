"""
Fetch MINiML XML for GEO accessions (GSM/GSE) directly from NCBI.

Used as the default data source for `meta_extract_*` when the caller does not
supply local mapping JSONs.
"""
import sys
import time
import xml.etree.ElementTree as ET

import requests

GEO_ACC_CGI = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
NS = {"ns": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
MIN_GEO_WAIT_TIME = 0.35
TIMEOUT = 15
MAX_TRIES = 3


def fetch_geo_xml(accession, targ="self", view="quick", verbose=False):
    """
    Fetch MINiML XML for a GEO accession into memory.

    Args:
        accession (str): GSM or GSE accession (e.g., "GSM534473", "GSE20752").
        targ (str): "self" (default), "series", or "all". Use "series" / "all"
            on a GSM to also retrieve parent GSE info.
        view (str): NCBI view parameter (default "quick").
        verbose (bool): If True, print retry warnings to stderr.

    Returns:
        str | None: The XML body as a string, or None if all retries failed.
            A "Could not fetch ..." line is always printed to stderr on final
            failure, regardless of verbose.
    """
    params = {"acc": accession, "view": view, "form": "xml", "targ": targ}
    last_err = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            time.sleep(MIN_GEO_WAIT_TIME)
            r = requests.get(GEO_ACC_CGI, params=params, timeout=TIMEOUT,
                             allow_redirects=True)
            r.encoding = "utf-8"
            if r.status_code == 200 and r.text and r.text.lstrip().startswith("<?xml"):
                return r.text
            last_err = f"HTTP {r.status_code}, body[:120]={r.text[:120]!r}"
        except requests.RequestException as e:
            last_err = repr(e)
        if verbose:
            print(
                f"  [retry {attempt}/{MAX_TRIES}] {accession} targ={targ}: {last_err}",
                file=sys.stderr,
            )
    print(
        f"Could not fetch {accession} (targ={targ}) after {MAX_TRIES} tries: {last_err}",
        file=sys.stderr,
    )
    return None


def find_iids(xml_str, local_name):
    """Return `iid` attribute values for elements with a given local tag name."""
    root = ET.fromstring(xml_str)
    out = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == local_name:
            iid = elem.attrib.get("iid")
            if iid:
                out.append(iid)
    return out


def find_sample_refs(xml_str):
    """Return `ref` attribute values from <Sample-Ref> elements (child GSM accessions of a GSE)."""
    root = ET.fromstring(xml_str)
    out = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Sample-Ref":
            ref = elem.attrib.get("ref")
            if ref:
                out.append(ref)
    return out


def discover_parent_gses(gsm_id, verbose=False):
    """
    Fetch a GSM's MINiML XML and discover its parent GSE accession(s).

    Tries `targ='series'` first; falls back to `targ='all'` if no Series iids
    are found. The returned GSM XML is suitable for `simplify_gsm_xml_file`.

    Args:
        gsm_id (str): GSM accession.
        verbose (bool): Pass-through to fetch_geo_xml retry warnings.

    Returns:
        tuple[str | None, list[str]]: (gsm_xml, parent_gse_ids).
            gsm_xml is None when the fetch fails entirely; parent_gse_ids is
            empty when no Series iids could be discovered.
    """
    gsm_xml = fetch_geo_xml(gsm_id, targ="series", verbose=verbose)
    if gsm_xml is not None:
        gses = find_iids(gsm_xml, "Series")
        if gses:
            return gsm_xml, gses
        # series fetch worked but didn't surface Series iids; try targ=all
        fallback = fetch_geo_xml(gsm_id, targ="all", verbose=verbose)
        if fallback is not None:
            gses = find_iids(fallback, "Series")
            return fallback, gses
        return gsm_xml, []
    # series fetch failed — try targ=all
    fallback = fetch_geo_xml(gsm_id, targ="all", verbose=verbose)
    if fallback is not None:
        return fallback, find_iids(fallback, "Series")
    # both failed — last try with targ=self so callers still get GSM content
    plain = fetch_geo_xml(gsm_id, targ="self", verbose=verbose)
    return plain, []


def expand_gse_to_gsms(gse_id, verbose=False):
    """
    Fetch a GSE and return its child GSM accessions (the <Sample-Ref ref="..."> values).

    Args:
        gse_id (str): GSE accession.
        verbose (bool): Pass-through to fetch_geo_xml retry warnings.

    Returns:
        list[str]: Child GSM accessions, or [] if the GSE could not be fetched.
    """
    xml_str = fetch_geo_xml(gse_id, targ="self", verbose=verbose)
    if xml_str is None:
        return []
    return find_sample_refs(xml_str)
