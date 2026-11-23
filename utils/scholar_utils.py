# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import re
import warnings
import os
import requests
import time
from typing import List, Dict, Union, Optional
from urllib.parse import quote
from thefuzz import fuzz  # type: ignore
import datetime

from dotenv import load_dotenv  # type: ignore

from utils import cli_backend

cur_dir = os.path.dirname(os.path.realpath(__file__))
load_dotenv(os.path.join(cur_dir, "../.env"))
S2_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
print(f"Using S2 API Key: {'yes' if S2_API_KEY else 'no'}")  # never log the key itself
OPENALEX_API_KEY = os.environ.get("OPENALEX_API_KEY")

_openalex_limit_warned = False


# --- OPENALEX FALLBACK (no API key) ---


def _rebuild_openalex_abstract(inverted_index: Optional[Dict]) -> Optional[str]:
    if not inverted_index:
        return None
    positions = [
        (position, word)
        for word, word_positions in inverted_index.items()
        for position in word_positions
    ]
    return " ".join(word for _, word in sorted(positions))


def _openalex_to_s2_shape(work: Dict) -> Dict:
    source = (work.get("primary_location") or {}).get("source") or {}
    biblio = work.get("biblio") or {}
    first_page, last_page = biblio.get("first_page"), biblio.get("last_page")
    pages = f"{first_page}-{last_page}" if first_page and last_page else first_page
    venue = source.get("display_name")
    return {
        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        "title": work.get("title"),
        "authors": [
            {"name": a["author"]["display_name"]}
            for a in work.get("authorships", [])
            if (a.get("author") or {}).get("display_name")
        ],
        "venue": venue,
        "year": work.get("publication_year"),
        "abstract": _rebuild_openalex_abstract(work.get("abstract_inverted_index")),
        "citationCount": work.get("cited_by_count"),
        "journal": {
            "name": venue,
            "volume": biblio.get("volume"),
            "issue": biblio.get("issue"),
            "pages": pages,
        },
        "publicationDate": work.get("publication_date"),
    }


def openalex_title_search(title_query: str) -> Optional[List[Dict]]:
    global _openalex_limit_warned
    try:
        rsp = requests.get(
            "https://api.openalex.org/works",
            # Sent as a header so the key never appears in a URL or error message.
            headers=(
                {"Authorization": f"Bearer {OPENALEX_API_KEY}"} if OPENALEX_API_KEY else {}
            ),
            params={
                "search": title_query,
                "per-page": 3,
                "select": "doi,title,publication_year,publication_date,cited_by_count,"
                "authorships,primary_location,abstract_inverted_index,biblio",
            },
            timeout=15,
        )
        if rsp.status_code == 429:
            if not _openalex_limit_warned:
                _openalex_limit_warned = True
                print(
                    "      ⚠️ OpenAlex daily allowance is used up. "
                    "Remaining candidate papers will be dropped."
                )
            return None
        if rsp.status_code != 200:
            return None
        return [_openalex_to_s2_shape(w) for w in rsp.json().get("results", [])]
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"      ⚠️ OpenAlex error for '{title_query}': {e}")
        return None


# --- CROSSREF FALLBACK (free, no API key) ---


def _crossref_to_s2_shape(item: Dict) -> Dict:
    date_parts = ((item.get("issued") or {}).get("date-parts") or [[]])[0]
    year = date_parts[0] if date_parts else None
    publication_date = None
    if len(date_parts) == 3:
        publication_date = "{:04d}-{:02d}-{:02d}".format(*date_parts)

    abstract = item.get("abstract")
    if abstract:  # Crossref abstracts are JATS XML
        abstract = html.unescape(re.sub(r"<[^>]+>", " ", abstract))
        abstract = re.sub(r"\s+", " ", abstract).strip()

    venue = (item.get("container-title") or [None])[0]
    venue = html.unescape(venue) if venue else venue
    title = (item.get("title") or [None])[0]
    subtitle = (item.get("subtitle") or [None])[0]
    if title and subtitle:  # Crossref stores the part after the colon separately
        title = f"{title}: {subtitle}"
    return {
        "doi": item.get("DOI"),
        "title": html.unescape(title) if title else title,
        "authors": [
            {"name": " ".join(filter(None, [a.get("given"), a.get("family")]))}
            for a in item.get("author", [])
            if a.get("family")
        ],
        "venue": venue,
        "year": year,
        "abstract": abstract,
        "citationCount": item.get("is-referenced-by-count"),
        "journal": {
            "name": venue,
            "volume": item.get("volume"),
            "issue": item.get("issue"),
            "pages": item.get("page"),
        },
        "publicationDate": publication_date,
    }


def _title_key(title: str) -> str:
    """Lowercase title without punctuation, so curly and straight quotes compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", html.unescape(title).lower())).strip()


def _title_ratio(query: str, title: str) -> int:
    """Compares titles; if only one side has a subtitle, compares the main titles."""
    ratio = fuzz.ratio(_title_key(query), _title_key(title))
    if (":" in query) != (":" in title):
        main_ratio = fuzz.ratio(
            _title_key(query.split(":")[0]), _title_key(title.split(":")[0])
        )
        ratio = max(ratio, main_ratio)
    return ratio


def _verify_with_crossref(paper: Dict) -> Optional[Dict]:
    """Takes bibliographic fields from the DOI's Crossref record; keeps the abstract."""
    doi = paper.get("doi") or (paper.get("externalIds") or {}).get("DOI")
    if doi:
        try:
            rsp = requests.get(
                f"https://api.crossref.org/works/{quote(doi, safe='/')}", timeout=15
            )
            if rsp.status_code == 200:
                record = _crossref_to_s2_shape(rsp.json()["message"])
                if record["title"] and paper.get("title"):
                    similarity = _title_ratio(record["title"], paper["title"])
                    if similarity < 85:
                        print(f"      ⚠️ DOI/title mismatch, dropped: '{paper['title']}'")
                        return None
                abstract = paper.get("abstract") or record.get("abstract")
                paper = {
                    **paper,
                    **{k: v for k, v in record.items() if v not in (None, "", [])},
                }
                paper["abstract"] = abstract
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            print(f"      ⚠️ Crossref DOI check failed for '{doi}': {e}")
    # Without a venue the pipeline would label the record "arXiv".
    return paper if paper.get("venue") else None


def crossref_title_search(title_query: str) -> Optional[List[Dict]]:
    try:
        rsp = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.bibliographic": title_query,
                "rows": 3,
                "select": "DOI,title,subtitle,author,container-title,issued,abstract,"
                "is-referenced-by-count,volume,issue,page",
            },
            timeout=15,
        )
        if rsp.status_code != 200:
            return None
        # A record without a venue would be labelled "arXiv" downstream, so skip it.
        return [
            _crossref_to_s2_shape(i)
            for i in rsp.json()["message"].get("items", [])
            if i.get("container-title")
        ]
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        print(f"      ⚠️ Crossref error for '{title_query}': {e}")
        return None


# --- SEMANTIC SCHOLAR UTILS ---


def s2_title_search(
    title_query: str, year_hint: int = None, cutoff_date: str = None
) -> Optional[Dict]:
    headers = {"X-API-KEY": S2_API_KEY} if S2_API_KEY else {}

    def is_date_valid(p_date, p_year, cutoff_str):
        if not cutoff_str:
            return True
        try:
            c_year, c_month = map(int, cutoff_str.split("-"))
            cutoff_dt = datetime.datetime(c_year, c_month, 1)
            if p_date:
                try:
                    p_dt = datetime.datetime.strptime(p_date, "%Y-%m-%d")
                    return p_dt < cutoff_dt
                except ValueError:
                    pass
            if p_year:
                if p_year < c_year:
                    return True
                if p_year == c_year and c_month > 1:
                    return True
                return False
            return True
        except Exception:
            return True

    try:
        results = None
        try:
            rsp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                headers=headers,
                params={
                    "query": title_query,
                    "limit": 3,
                    "fields": "title,authors,venue,year,abstract,citationCount,journal,publicationDate,externalIds",
                },
                timeout=5,
            )
            if rsp.status_code == 200:
                results = rsp.json().get("data", [])
        except requests.RequestException as e:
            print(f"      ⚠️ S2 request failed for '{title_query}': {e}")

        use_cli = cli_backend.active_backend() is not None

        def pick_best(candidates):
            best_match = None
            highest_ratio = 0
            for r in candidates or []:
                if not r.get("title"):
                    continue
                if not is_date_valid(r.get("publicationDate"), r.get("year"), cutoff_date):
                    continue

                if use_cli:
                    ratio = _title_ratio(title_query, r["title"])
                else:
                    ratio = fuzz.ratio(title_query.lower(), r["title"].lower())
                    if year_hint and r.get("year") == year_hint:
                        ratio += 10

                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = r
            return best_match, highest_ratio

        if not use_cli:
            if not results:
                return None
            best_match, highest_ratio = pick_best(results)
            return best_match if highest_ratio > 70 else None

        # CLI runs need an exact title match, a verified record and an abstract.
        # At the original threshold of 70, different papers with similar titles passed.
        sources = (
            lambda: results,
            lambda: openalex_title_search(title_query),
            lambda: crossref_title_search(title_query),
        )
        for source in sources:
            best_match, highest_ratio = pick_best(source())
            if best_match and highest_ratio >= 95:
                verified = _verify_with_crossref(best_match)
                if verified and verified.get("abstract"):
                    return verified
        return None
    except Exception as e:
        print(f"      ⚠️ S2 Error for '{title_query}': {e}")
        return None
