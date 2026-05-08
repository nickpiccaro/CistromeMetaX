#!/usr/bin/env python3
"""
Smoke test: verify we can fetch GSM and GSE MINiML XMLs from NCBI GEO
into memory and connect a GSM to its parent GSE(s) and a GSE to its
child GSM(s).

Run:
    python tests/scratch/smoke_geo_fetch.py
    python tests/scratch/smoke_geo_fetch.py --dump   # also write XMLs to tests/scratch/_out/

This is NOT part of the pytest suite. It hits the live NCBI endpoint.
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import requests

GEO_ACC_CGI = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
NS = {"ns": "http://www.ncbi.nlm.nih.gov/geo/info/MINiML"}
MIN_GEO_WAIT_TIME = 0.35
TIMEOUT = 15
MAX_TRIES = 3

# Probe accession from this repo's sample.xml
PROBE_GSM = "GSM534473"


def fetch_geo_xml(accession: str, targ: str = "self", view: str = "quick") -> str:
    """Fetch MINiML XML for a GEO accession into memory. Returns the body as a string."""
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
        print(f"  [retry {attempt}/{MAX_TRIES}] {accession} targ={targ}: {last_err}")
    raise RuntimeError(f"Failed to fetch {accession} (targ={targ}): {last_err}")


def find_iids(xml_str: str, local_name: str):
    """Return a list of `iid` attribute values for elements with a given local name (namespace-agnostic)."""
    root = ET.fromstring(xml_str)
    out = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == local_name:
            iid = elem.attrib.get("iid")
            if iid:
                out.append(iid)
    return out


def find_sample_refs(xml_str: str):
    """Return values of `ref` attribute on <Sample-Ref> elements."""
    root = ET.fromstring(xml_str)
    out = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "Sample-Ref":
            ref = elem.attrib.get("ref")
            if ref:
                out.append(ref)
    return out


# ---- Adapted simplifiers (string-based versions of the package functions) ----
# These are inline copies of CistromeMetaX.parser_extractor.simplify_gsm_xml_file
# and simplify_gse_xml_file, modified to take an XML string via ET.fromstring().
# We're NOT modifying the package — just verifying the same logic works on the
# live NCBI document.

def simplify_gsm_xml_string(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    sample = root.find("ns:Sample", NS)
    if sample is None:
        return "Error: <Sample> element not found in XML."

    extracted_data = {}
    exclude_elements = {
        "Platform", "Platform-Ref", "Library-Source", "Library-Selection",
        "Instrument-Model", "Contact-Ref", "Supplementary-Data", "Relation",
        "Data-Processing", "Extract-Protocol",
    }
    channel_data = []

    for child in sample:
        tag = child.tag.split("}")[-1]
        if tag in exclude_elements:
            continue
        if tag == "Channel":
            channel_content = []
            for subchild in child:
                subtag = subchild.tag.split("}")[-1]
                if subtag == "Extract-Protocol":
                    continue
                attributes = " ".join(f'({k}="{v}")' for k, v in subchild.attrib.items())
                value = f"{subtag}{attributes}: {subchild.text.strip()}" if subchild.text else None
                if value:
                    channel_content.append(value)
            if channel_content:
                channel_data.append("\n".join(channel_content))
        else:
            attributes = " ".join(f'({k}="{v}")' for k, v in child.attrib.items())
            value = f"{tag}{attributes}: {child.text.strip()}" if child.text else None
            if value:
                extracted_data[tag] = value

    output_lines = list(extracted_data.values())
    if channel_data:
        output_lines.append("Channel:")
        output_lines.append("\n".join(channel_data))
    return "\n".join(output_lines)


def simplify_gse_xml_string(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    series = root.find("ns:Series", NS)
    if series is None:
        return "Error: <Series> element not found in XML."

    exclude_elements = {"Status", "Contributor-Ref", "Sample-Ref", "Relation", "Supplementary-Data"}
    extracted_data = {}
    for child in series:
        tag = child.tag.split("}")[-1]
        if tag in exclude_elements:
            continue
        attributes = " ".join(f'({k}="{v}")' for k, v in child.attrib.items())
        value = f"{tag}{attributes}: {child.text.strip()}" if child.text else None
        if value:
            extracted_data[tag] = value

    return "\n".join(extracted_data.values())


# ---- Driver ----

def banner(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def maybe_dump(out_dir, name, content):
    if out_dir is None:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)
    print(f"  [dump] wrote {path} ({len(content)} chars)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm", default=PROBE_GSM, help=f"GSM probe accession (default: {PROBE_GSM})")
    parser.add_argument("--dump", action="store_true",
                        help="write fetched XMLs to tests/scratch/_out/")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out") if args.dump else None

    # ---- Step 1: GSM fetch under three targ values ----
    banner(f"Step 1: fetch GSM {args.gsm} with targ in (self, series, all)")
    targ_results = {}
    for targ in ("self", "series", "all"):
        try:
            xml_str = fetch_geo_xml(args.gsm, targ=targ)
        except RuntimeError as e:
            print(f"  targ={targ}: FAILED -> {e}")
            continue
        series_iids = find_iids(xml_str, "Series")
        sample_iids = find_iids(xml_str, "Sample")
        sample_refs = find_sample_refs(xml_str)
        print(f"  targ={targ:<6} | size={len(xml_str):>7} chars | "
              f"Series iids={series_iids} | Sample iids count={len(sample_iids)} | "
              f"Sample-Ref count={len(sample_refs)}")
        if sample_iids and len(sample_iids) <= 5:
            print(f"            sample iids -> {sample_iids}")
        targ_results[targ] = {
            "xml": xml_str,
            "series_iids": series_iids,
            "sample_iids": sample_iids,
            "sample_refs": sample_refs,
        }
        maybe_dump(out_dir, f"{args.gsm}_targ-{targ}.xml", xml_str)

    # ---- Step 2: pick a parent GSE ----
    banner("Step 2: identify parent GSE for the probe GSM")
    parent_gses = []
    for targ in ("series", "all", "self"):
        if targ in targ_results and targ_results[targ]["series_iids"]:
            parent_gses = targ_results[targ]["series_iids"]
            print(f"  parent GSE(s) discovered via targ={targ}: {parent_gses}")
            break
    if not parent_gses:
        print("  ERROR: no parent GSE could be identified from any targ value.")
        sys.exit(1)
    gse_id = parent_gses[0]

    # ---- Step 3: fetch the GSE and enumerate child GSMs ----
    banner(f"Step 3: fetch GSE {gse_id} with targ=self and list child GSMs")
    gse_xml = fetch_geo_xml(gse_id, targ="self")
    maybe_dump(out_dir, f"{gse_id}_targ-self.xml", gse_xml)
    series_iids = find_iids(gse_xml, "Series")
    child_refs = find_sample_refs(gse_xml)
    print(f"  GSE size={len(gse_xml)} chars | Series iids={series_iids} | "
          f"child Sample-Ref count={len(child_refs)}")
    print(f"  first 5 child GSMs: {child_refs[:5]}")
    if args.gsm in child_refs:
        print(f"  OK: probe GSM {args.gsm} appears in GSE child Sample-Refs")
    else:
        print(f"  WARN: probe GSM {args.gsm} NOT found in GSE child Sample-Refs")

    # ---- Step 4: round-trip simplifiers on fetched strings ----
    banner("Step 4: round-trip fetched XMLs through string-based simplifiers")
    gsm_xml = targ_results["self"]["xml"] if "self" in targ_results else next(iter(targ_results.values()))["xml"]
    print("\n--- simplify_gsm_xml_string output ---")
    print(simplify_gsm_xml_string(gsm_xml))
    print("\n--- simplify_gse_xml_string output ---")
    print(simplify_gse_xml_string(gse_xml))

    banner("DONE")
    print("If output above looks reasonable, the fetch+parse approach is viable.")


if __name__ == "__main__":
    main()
