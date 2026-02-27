from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

import requests
from PyPDF2 import PdfReader


DEFAULT_MCP_URL = "https://ftec5660.ngrok.app/mcp"


def _now_year() -> int:
    return datetime.now().year


def _clamp01(x: float) -> float:
    if x < 0:
        return 0.0
    if x > 1:
        return 1.0
    return float(x)


def _norm_space(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_camel(s: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s)


def _fix_dashes(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("鈥?", "-").replace("–", "-")
    return s


def _clean_text(s: str) -> str:
    s = _split_camel(s)
    s = _fix_dashes(s)
    s = re.sub(r"[\u2022\u25cf\u25aa\u25a0\u25e6]", " ", s)
    s = re.sub(r"[^\S\r\n]+", " ", s)
    return _norm_space(s)


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts: list[str] = []
    for p in reader.pages:
        parts.append(p.extract_text() or "")
    return _clean_text("\n".join(parts))


def _lines(text: str) -> list[str]:
    raw = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in raw if ln]


def _similar(a: str, b: str) -> float:
    a = re.sub(r"\s+", " ", a.strip().lower())
    b = re.sub(r"\s+", " ", b.strip().lower())
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def _tok(s: str) -> set[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    words = [w for w in s.split() if len(w) >= 2]
    return set(words)


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _extract_name(lines: list[str]) -> str:
    if not lines:
        return ""
    head = lines[0]
    orig = head
    head = re.sub(r"\s*[,|/].*$", "", head).strip()
    m = re.match(r"^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})", head)
    if m:
        name = m.group(1).strip()
        if "," in orig:
            parts = name.split()
            if len(parts) >= 3:
                name = " ".join(parts[:2])
        return name
    m = re.match(r"^([A-Za-z]+(?:\s+[A-Za-z]+){0,3})", head)
    return (m.group(1).strip() if m else head.strip())[:60]


def _extract_locations(lines: list[str]) -> list[str]:
    return _extract_locations_with_name(lines, name=None)


def _extract_locations_with_name(lines: list[str], name: Optional[str]) -> list[str]:
    def strip_name_prefix(seg: str) -> str:
        if not name:
            return seg
        n = name.strip()
        if not n:
            return seg
        s = seg.strip()
        if s.lower().startswith(n.lower()):
            s = s[len(n) :].strip(" ,|-")
        parts = n.split()
        if len(parts) >= 2:
            short = f"{parts[0]} {parts[-1]}".strip()
            if s.lower().startswith(short.lower()):
                s = s[len(short) :].strip(" ,|-")
        return s

    bad_kw = {
        "experience",
        "education",
        "skills",
        "professional",
        "engineer",
        "manager",
        "analyst",
        "consultant",
        "scientist",
        "worked",
        "managed",
        "designed",
        "supported",
        "advised",
        "led",
        "current",
        "present",
        "graduated",
    }

    def looks_like_place(seg: str) -> Optional[str]:
        s = seg.strip()
        s = re.sub(r"^\W+", "", s)
        s = re.sub(r"\(Hometown\)", "", s, flags=re.IGNORECASE).strip()
        s = strip_name_prefix(s)
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^(alt|home|h|location)\s+", "", s, flags=re.IGNORECASE).strip()
        if not s or len(s) > 40:
            return None
        if any(k in s.lower() for k in bad_kw):
            return None
        if re.search(r"\d", s):
            return None
        if s.endswith("."):
            return None
        if "," in s:
            a, b = [x.strip() for x in s.split(",", 1)]
            if not a or not b:
                return None
            if a.isupper() or b.isupper():
                return None
            if not re.search(r"[A-Za-z]", a) or not re.search(r"[A-Za-z]", b):
                return None
            return _norm_space(f"{a}, {b}")
        words = [w for w in s.split() if w]
        if not (1 <= len(words) <= 3):
            return None
        if not all(re.match(r"^[A-Z][A-Za-z.'-]*$", w) for w in words):
            return None
        return _norm_space(s)

    out: list[str] = []
    for ln in lines[:10]:
        if not any(x in ln for x in [",", "|", "/"]):
            continue
        parts = re.split(r"[|/]+", ln)
        for p in parts:
            cand = looks_like_place(p)
            if cand:
                out.append(cand)

    uniq: list[str] = []
    seen: set[str] = set()
    for x in out:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq[:6]


def _extract_skills(lines: list[str]) -> list[str]:
    joined = "\n".join(lines)
    m = re.search(r"(?:^|\n)(?:Skills|Core Skills)\s*\n(.+)", joined, flags=re.IGNORECASE)
    if not m:
        return []
    tail = m.group(1)
    tail = tail.split("\nEducation\n", 1)[0]
    tail = tail.split("\nExperience\n", 1)[0]
    tail = tail.split("\nProfessional Experience\n", 1)[0]
    tail = tail.replace(",", " ")
    tail = re.sub(r"[^\w\s&+/.-]", " ", tail)
    words = [w.strip() for w in re.split(r"\s+", tail) if w.strip()]
    skills: list[str] = []
    buf: list[str] = []
    for w in words:
        if len(w) <= 1:
            continue
        if w.lower() in {"skills", "education", "experience", "professional", "core", "frameworks", "tools"}:
            continue
        if re.fullmatch(r"\d{1,2}", w):
            continue
        buf.append(w)
    merged = " ".join(buf)
    for seg in re.split(r"\s{2,}|/|;|\n", merged):
        seg = _norm_space(seg)
        if not seg:
            continue
        if len(seg) > 40:
            for part in seg.split():
                if 2 <= len(part) <= 24:
                    skills.append(part)
        else:
            skills.append(seg)
    uniq: list[str] = []
    seen: set[str] = set()
    for s in skills:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq[:30]


def _extract_education(lines: list[str]) -> list[dict[str, Any]]:
    edu: list[dict[str, Any]] = []
    for i, ln in enumerate(lines):
        if re.search(r"\b(University|College|School|Institute)\b", ln, flags=re.IGNORECASE) or re.search(
            r"\b(BSc|MSc|MBA|PhD|Bachelor|Master)\b", ln, flags=re.IGNORECASE
        ):
            window = " ".join(lines[i : i + 3])
            years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", window)]
            school = ""
            ms = re.search(r"([A-Za-z][A-Za-z .&-]{2,}(?:University|College|School|Institute)[A-Za-z .&-]{0,30})", window)
            if ms:
                school = _norm_space(ms.group(1))
            degree = ""
            md = re.search(r"\b(BSc|MSc|MBA|PhD|Bachelor of [A-Za-z ]+|Master of [A-Za-z ]+)\b", window, flags=re.IGNORECASE)
            if md:
                degree = _norm_space(md.group(0))
            item: dict[str, Any] = {}
            if school:
                item["school"] = school
            if degree:
                item["degree"] = degree
            if years:
                item["years"] = sorted(set(years))
            if item:
                edu.append(item)
    uniq: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in edu:
        k = json.dumps(e, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq[:6]


def _extract_experience(lines: list[str]) -> list[dict[str, Any]]:
    exps: list[dict[str, Any]] = []

    def parse_year_token(tok: str) -> Optional[int]:
        if tok.lower() in {"present", "current"}:
            return None
        m = re.fullmatch(r"(19\d{2}|20\d{2})", tok)
        return int(m.group(1)) if m else None

    for ln in lines:
        ln2 = re.sub(r"([A-Za-z])(\d{4})", r"\1 \2", ln)
        ln2 = re.sub(r"(\d{4})([A-Za-z])", r"\1 \2", ln2)
        m = re.search(r"\b(19\d{2}|20\d{2})\b\s*-\s*\b(Present|Current|19\d{2}|20\d{2})\b", ln2, flags=re.IGNORECASE)
        if not m:
            continue
        start_year = int(m.group(1))
        end_tok = m.group(2)
        end_year = parse_year_token(end_tok)
        is_current = end_year is None
        left = ln2[: m.start()].strip()
        right = ln2[m.end() :].strip()
        title = ""
        company = ""
        tail = (right or left).strip()
        tail = re.sub(r"^\W+", "", tail)
        if "," in tail:
            a, b = tail.split(",", 1)
            title = _norm_space(a)
            company = _norm_space(b)
        elif " at " in tail.lower():
            parts = re.split(r"\bat\b", tail, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                title = _norm_space(parts[0])
                company = _norm_space(parts[1])
        else:
            if left and right:
                tail2 = f"{left} {right}".strip()
            else:
                tail2 = tail
            chunks = [c.strip() for c in re.split(r"\s{2,}|\t", tail2) if c.strip()]
            if len(chunks) >= 2:
                title = chunks[0]
                company = chunks[1]
            else:
                company = tail2
        item: dict[str, Any] = {
            "start_year": start_year,
            "end_year": end_year,
            "is_current": bool(is_current),
        }
        if title:
            item["title"] = title[:80]
        if company:
            item["company"] = company[:80]
        exps.append(item)

    uniq: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in exps:
        k = json.dumps(e, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq[:12]


@dataclass
class CVData:
    path: str
    text: str
    name: str
    locations: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    education: list[dict[str, Any]] = field(default_factory=list)
    experience: list[dict[str, Any]] = field(default_factory=list)


def parse_cv(path: str) -> CVData:
    text = extract_pdf_text(path)
    ls = _lines(text)
    name = _extract_name(ls)
    locations = _extract_locations_with_name(ls, name=name)
    skills = _extract_skills(ls)
    education = _extract_education(ls)
    experience = _extract_experience(ls)
    return CVData(
        path=path,
        text=text,
        name=name,
        locations=locations,
        skills=skills,
        education=education,
        experience=experience,
    )


class MCPHttpClient:
    def __init__(self, url: str, headers: Optional[dict[str, str]] = None, timeout_s: float = 30.0):
        self.url = url
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.base_headers = dict(headers or {})
        self.base_headers.setdefault("Accept", "application/json, text/event-stream")
        self._next_id = 1
        self._session_id: Optional[str] = None
        self._tool_name_map: dict[str, str] = {}

    def _headers(self) -> dict[str, str]:
        h = dict(self.base_headers)
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    @staticmethod
    def _parse_sse_messages(text: str) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip("\r")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if isinstance(obj, dict):
                msgs.append(obj)
        return msgs

    def _decode_response(self, resp: requests.Response, request_id: Optional[int]) -> Any:
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "text/event-stream" in ctype or resp.text.lstrip().startswith("event:") or "\ndata:" in resp.text:
            msgs = self._parse_sse_messages(resp.text)
            if request_id is None:
                return msgs[-1] if msgs else {}
            for m in msgs:
                if m.get("id") == request_id:
                    return m
            return msgs[-1] if msgs else {}
        return resp.json()

    def _rpc(self, method: str, params: Optional[dict[str, Any]] = None) -> Any:
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        request_id = self._next_id
        self._next_id += 1
        if params is not None:
            payload["params"] = params
        resp = self.session.post(self.url, json=payload, headers=self._headers(), timeout=self.timeout_s)
        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers.get("mcp-session-id") or self._session_id
        if "Mcp-Session-Id" in resp.headers:
            self._session_id = resp.headers.get("Mcp-Session-Id") or self._session_id
        resp.raise_for_status()
        data = self._decode_response(resp, request_id)
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(str(data["error"]))
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    def _notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        resp = self.session.post(self.url, json=payload, headers=self._headers(), timeout=self.timeout_s)
        if "mcp-session-id" in resp.headers:
            self._session_id = resp.headers.get("mcp-session-id") or self._session_id
        if "Mcp-Session-Id" in resp.headers:
            self._session_id = resp.headers.get("Mcp-Session-Id") or self._session_id
        if resp.status_code >= 400:
            resp.raise_for_status()

    def initialize(self) -> None:
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hw2-cv-verifier", "version": "1.0.0"},
            },
        )
        try:
            self._notify("notifications/initialized", {})
        except Exception:
            pass

    @staticmethod
    def _norm_tool_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def tools_list(self) -> list[dict[str, Any]]:
        res = self._rpc("tools/list", None)
        if isinstance(res, dict) and "tools" in res and isinstance(res["tools"], list):
            tools = res["tools"]
        elif isinstance(res, list):
            tools = res
        else:
            tools = []
        out: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict) and "name" in t:
                out.append(t)
        return out

    def refresh_tool_map(self) -> None:
        mapping: dict[str, str] = {}
        for t in self.tools_list():
            name = str(t.get("name", ""))
            if name:
                mapping[self._norm_tool_name(name)] = name
        self._tool_name_map = mapping

    def resolve_tool(self, desired: str) -> str:
        if not self._tool_name_map:
            self.refresh_tool_map()
        k = self._norm_tool_name(desired)
        if k in self._tool_name_map:
            return self._tool_name_map[k]
        for nk, actual in self._tool_name_map.items():
            if nk.endswith(k) or k.endswith(nk) or k in nk or nk in k:
                return actual
        return desired

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        actual = self.resolve_tool(tool_name)
        res = self._rpc("tools/call", {"name": actual, "arguments": arguments})
        return self._unwrap_tool_result(res)

    @staticmethod
    def _unwrap_tool_result(res: Any) -> Any:
        if isinstance(res, dict) and "content" in res and isinstance(res["content"], list):
            content = res["content"]
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "json" and "json" in item:
                    return item["json"]
                if item.get("type") == "text" and "text" in item:
                    txt = item["text"]
                    try:
                        return json.loads(txt)
                    except Exception:
                        return txt
            return content
        if isinstance(res, dict) and "structuredContent" in res:
            return res["structuredContent"]
        return res


def _best_location_hint(locations: list[str]) -> Optional[str]:
    if not locations:
        return None
    for loc in locations:
        if "," in loc:
            city, country = [x.strip() for x in loc.split(",", 1)]
            if country:
                return country
            if city:
                return city
    return locations[0].strip() or None


def _cv_keywords(cv: CVData) -> set[str]:
    base = set()
    base |= _tok(cv.name)
    for loc in cv.locations:
        base |= _tok(loc)
    for s in cv.skills:
        base |= _tok(s)
    for e in cv.experience:
        if "company" in e:
            base |= _tok(str(e["company"]))
        if "title" in e:
            base |= _tok(str(e["title"]))
    for edu in cv.education:
        for v in edu.values():
            if isinstance(v, str):
                base |= _tok(v)
    stop = {"professional", "experience", "skills", "education", "engineer", "consulting", "consultant"}
    return {w for w in base if w not in stop}


def _score_linkedin_result(cv: CVData, r: dict[str, Any]) -> float:
    s = 0.0
    name = str(r.get("name", ""))
    loc = str(r.get("location", ""))
    headline = str(r.get("headline", ""))
    match_type = str(r.get("match_type", ""))
    s += 2.0 * _similar(cv.name, name)
    if match_type == "exact":
        s += 0.8
    hint = _best_location_hint(cv.locations) or ""
    if hint:
        s += 0.8 * _similar(hint, loc)
    cvk = _cv_keywords(cv)
    s += 0.8 * _jaccard(cvk, _tok(headline))
    return float(s)


def _choose_best(items: list[dict[str, Any]], score_fn) -> Optional[dict[str, Any]]:
    if not items:
        return None
    best = None
    best_s = None
    for it in items:
        try:
            sc = float(score_fn(it))
        except Exception:
            continue
        if best_s is None or sc > best_s:
            best_s = sc
            best = it
    return best


def _exc_brief(e: BaseException, limit: int = 180) -> str:
    msg = str(e) if e is not None else ""
    msg = re.sub(r"\s+", " ", msg).strip()
    if len(msg) > limit:
        msg = msg[:limit]
    return f"{type(e).__name__}:{msg}" if msg else type(e).__name__


def _verify_consistency(cv: CVData, linkedin_profile: Optional[dict[str, Any]], facebook_profile: Optional[dict[str, Any]]) -> tuple[float, list[str], dict[str, Any]]:
    issues: list[str] = []
    nowy = _now_year()
    penalty = 0.0

    future_years: list[int] = []
    for edu in cv.education:
        ys = edu.get("years")
        if isinstance(ys, list):
            for y in ys:
                if isinstance(y, int) and y > nowy:
                    future_years.append(y)
    for ex in cv.experience:
        sy = ex.get("start_year")
        ey = ex.get("end_year")
        if isinstance(sy, int) and sy > nowy:
            future_years.append(sy)
        if isinstance(ey, int) and ey > nowy:
            future_years.append(ey)
    if future_years:
        issues.append(f"future_years:{sorted(set(future_years))}")
        penalty += 0.65

    if linkedin_profile:
        ln_name = str(linkedin_profile.get("name", ""))
        if ln_name and _similar(cv.name, ln_name) < 0.72:
            issues.append("linkedin_name_mismatch")
            penalty += 0.25
        ln_city = str(linkedin_profile.get("city", ""))
        ln_country = str(linkedin_profile.get("country", ""))
        if (ln_city or ln_country) and cv.locations:
            hint = _best_location_hint(cv.locations) or ""
            if hint and _similar(hint, f"{ln_city}, {ln_country}".strip(", ")) < 0.45:
                issues.append("linkedin_location_mismatch")
                penalty += 0.12

        cv_companies = [str(e.get("company", "")) for e in cv.experience if e.get("company")]
        ln_companies: list[str] = []
        for e in linkedin_profile.get("experience", []) or []:
            if isinstance(e, dict) and e.get("company"):
                ln_companies.append(str(e.get("company")))
        if cv_companies and ln_companies:
            cvset = {_norm_space(c).lower() for c in cv_companies if c.strip()}
            lnset = {_norm_space(c).lower() for c in ln_companies if c.strip()}
            matched = 0
            for c in cvset:
                if any(_similar(c, d) >= 0.8 for d in lnset):
                    matched += 1
            coverage = matched / max(1, len(cvset))
            if coverage < 0.34:
                issues.append("linkedin_experience_low_coverage")
                penalty += 0.24
            elif coverage < 0.5:
                issues.append("linkedin_experience_partial_coverage")
                penalty += 0.12

        cv_schools: list[str] = []
        for e in cv.education:
            if isinstance(e, dict) and "school" in e:
                cv_schools.append(str(e.get("school", "")))
        ln_schools: list[str] = []
        for e in linkedin_profile.get("education", []) or []:
            if isinstance(e, dict) and e.get("school"):
                ln_schools.append(str(e.get("school")))
        if cv_schools and ln_schools:
            cvset = {_norm_space(s).lower() for s in cv_schools if s.strip()}
            lnset = {_norm_space(s).lower() for s in ln_schools if s.strip()}
            matched = 0
            for s in cvset:
                if any(_similar(s, t) >= 0.8 for t in lnset):
                    matched += 1
            coverage = matched / max(1, len(cvset))
            if coverage < 0.34:
                issues.append("linkedin_education_low_coverage")
                penalty += 0.16

    if facebook_profile:
        fb_display = str(facebook_profile.get("display_name", ""))
        fb_original = str(facebook_profile.get("original_name", ""))
        if fb_display and _similar(cv.name, fb_display) < 0.6 and fb_original and _similar(cv.name, fb_original) < 0.6:
            issues.append("facebook_name_mismatch")
            penalty += 0.18
        fb_city = str(facebook_profile.get("city", ""))
        fb_country = str(facebook_profile.get("country", ""))
        if fb_city or fb_country:
            fb_loc = f"{fb_city}, {fb_country}".strip(", ")
            if cv.locations:
                best = max((_similar(loc, fb_loc) for loc in cv.locations), default=0.0)
                if best < 0.45:
                    issues.append("facebook_location_mismatch")
                    penalty += 0.1

        fb_company = str(facebook_profile.get("current_company", "") or "")
        if fb_company:
            cv_companies = [str(e.get("company", "")) for e in cv.experience if e.get("company")]
            if cv_companies:
                best = max((_similar(fb_company, c) for c in cv_companies), default=0.0)
                if best < 0.55:
                    issues.append("facebook_current_company_mismatch")
                    penalty += 0.12

    current_roles = [e for e in cv.experience if e.get("is_current")]
    if len(current_roles) >= 3:
        issues.append("too_many_current_roles")
        penalty += 0.22

    overlap_flag = False
    spans: list[tuple[int, int]] = []
    for e in cv.experience:
        sy = e.get("start_year")
        ey = e.get("end_year")
        if isinstance(sy, int):
            eyy = ey if isinstance(ey, int) else nowy
            spans.append((sy, eyy))
    spans.sort()
    for i in range(1, len(spans)):
        if spans[i][0] <= spans[i - 1][1] and spans[i][0] != spans[i - 1][0]:
            overlap_flag = True
            break
    if overlap_flag and len(spans) >= 3:
        issues.append("overlapping_experience_years")
        penalty += 0.1

    meta: dict[str, Any] = {"now_year": nowy}
    return penalty, issues, meta


def verify_cv(
    path: str,
    mcp_url: Optional[str] = None,
    mcp_headers: Optional[dict[str, str]] = None,
    timeout_s: float = 30.0,
    retry: int = 1,
) -> tuple[float, dict[str, Any]]:
    cv = parse_cv(path)
    report: dict[str, Any] = {
        "cv_path": path,
        "extracted": {
            "name": cv.name,
            "locations": cv.locations,
            "skills": cv.skills,
            "education": cv.education,
            "experience": cv.experience,
        },
        "matches": {},
        "issues": [],
    }

    linkedin_profile = None
    facebook_profile = None
    tool_errors: list[str] = []

    if mcp_url:
        client = MCPHttpClient(mcp_url, headers=mcp_headers, timeout_s=timeout_s)
        try:
            client.initialize()
        except Exception as e:
            tool_errors.append(f"initialize:{_exc_brief(e)}")
            client = None
        for attempt in range(retry + 1):
            try:
                if client is None:
                    break
                hint = _best_location_hint(cv.locations)
                ln_args: dict[str, Any] = {"q": cv.name, "limit": 10, "fuzzy": True}
                if hint:
                    ln_args["location"] = hint
                ln_results = client.call_tool("search_linkedin_people", ln_args)
                if isinstance(ln_results, dict) and "results" in ln_results:
                    ln_results = ln_results["results"]
                if not isinstance(ln_results, list):
                    ln_results = []
                best_ln = _choose_best(
                    [x for x in ln_results if isinstance(x, dict)],
                    lambda r: _score_linkedin_result(cv, r),
                )
                if best_ln and "id" in best_ln:
                    linkedin_profile = client.call_tool("get_linkedin_profile", {"person_id": int(best_ln["id"])})
                    report["matches"]["linkedin"] = {"search_result": best_ln, "profile": linkedin_profile}
                    try:
                        interactions = client.call_tool("get_linkedin_interactions", {"person_id": int(best_ln["id"])})
                        report["matches"]["linkedin"]["interactions"] = interactions
                    except Exception as e:
                        tool_errors.append(f"get_linkedin_interactions:{type(e).__name__}")
                else:
                    report["matches"]["linkedin"] = {"search_result": best_ln, "profile": None}

                fb_results = client.call_tool("search_facebook_users", {"q": cv.name, "limit": 10, "fuzzy": True})
                if isinstance(fb_results, dict) and "results" in fb_results:
                    fb_results = fb_results["results"]
                if not isinstance(fb_results, list):
                    fb_results = []
                hint = _best_location_hint(cv.locations) or ""

                def fb_score(r: dict[str, Any]) -> float:
                    s = 0.0
                    s += 2.0 * _similar(cv.name, str(r.get("display_name", "")))
                    if hint:
                        s += 0.8 * _similar(hint, f"{r.get('city','')}, {r.get('country','')}".strip(", "))
                    if str(r.get("match_type", "")) == "exact":
                        s += 0.5
                    return float(s)

                best_fb = _choose_best([x for x in fb_results if isinstance(x, dict)], fb_score)
                if best_fb and "id" in best_fb:
                    facebook_profile = client.call_tool("get_facebook_profile", {"user_id": int(best_fb["id"])})
                    report["matches"]["facebook"] = {"search_result": best_fb, "profile": facebook_profile}
                else:
                    report["matches"]["facebook"] = {"search_result": best_fb, "profile": None}

                break
            except Exception as e:
                tool_errors.append(f"mcp_attempt_{attempt}:{_exc_brief(e)}")
                if attempt < retry:
                    time.sleep(0.5 * (attempt + 1))
                else:
                    break

    base = 0.88
    if not mcp_url:
        base = 0.7
    if mcp_url and not linkedin_profile and not facebook_profile:
        base = 0.65 if tool_errors else 0.45

    penalty, issues, meta = _verify_consistency(cv, linkedin_profile, facebook_profile)
    if tool_errors:
        report["tool_errors"] = tool_errors

    if linkedin_profile:
        interactions = report.get("matches", {}).get("linkedin", {}).get("interactions")
        if isinstance(interactions, dict):
            es = interactions.get("engagement_score")
            pc = interactions.get("post_count")
            if isinstance(es, (int, float)) and isinstance(pc, int):
                if pc >= 5 and es >= 2:
                    base += 0.05
                elif pc == 0:
                    penalty += 0.04

    report["issues"] = issues
    report["meta"] = meta
    score = _clamp01(base - penalty)
    report["score"] = score
    return score, report


def score_cvs(
    cv_paths: Optional[list[str]] = None,
    mcp_url: Optional[str] = None,
    timeout_s: float = 30.0,
    retry: int = 1,
    write_report_path: Optional[str] = "verification_report.json",
) -> list[float]:
    if cv_paths is None:
        cv_paths = [f"CV_{i}.pdf" for i in range(1, 6)]
    headers = {"ngrok-skip-browser-warning": "true"} if mcp_url else None
    scores: list[float] = []
    reports: list[dict[str, Any]] = []
    for p in cv_paths:
        s, r = verify_cv(p, mcp_url=mcp_url, mcp_headers=headers, timeout_s=timeout_s, retry=retry)
        scores.append(float(s))
        reports.append(r)
    if write_report_path:
        with open(write_report_path, "w", encoding="utf-8") as f:
            json.dump({"scores": scores, "reports": reports}, f, ensure_ascii=False, indent=2)
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--mcp-url", default=os.environ.get("MCP_URL") or DEFAULT_MCP_URL)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retry", type=int, default=1)
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args()

    mcp_url = None if args.offline else (args.mcp_url.strip() or None)
    paths = args.paths or None
    report_path = None if args.no_report else "verification_report.json"
    scores = score_cvs(cv_paths=paths, mcp_url=mcp_url, timeout_s=args.timeout, retry=args.retry, write_report_path=report_path)
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
