import re
import json
import random
import hashlib
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_SKIP_TAGS = {"header", "footer", "aside", "nav"}
_SKIP_ATTRS = re.compile(
    r"\b(nav|footer|header|menu|breadcrumb|sidebar|filter|facet|pagination|"
    r"cart|checkout|newsletter|popup|modal|cookie|banner|subscribe|"
    r"social|share|wishlist-modal|refinement)\b",
    re.IGNORECASE,
)

_BAD_TEXT_RE = re.compile(
    r"\b(cookie|privacy|policy|terms|sign in|log in|subscribe|newsletter|"
    r"javascript|wishlist|compare|share|sort by|filter by|view more|load more)\b",
    re.IGNORECASE,
)

_CARD_PATTERN = re.compile(
    r"\b(product[-_]?card|product[-_]?tile|product[-_]?item|"
    r"grid[-_]?item|collection[-_]?item|plp[-_]item|"
    r"ProductCard|productCard)\b",
    re.IGNORECASE,
)

_PRODUCT_URL_PATTERNS = [
    re.compile(r"/[A-Z0-9][\w-]*\.html$", re.IGNORECASE),
    re.compile(r"/products/[\w-]+$", re.IGNORECASE),
    re.compile(r"/product/[\w-]+/?$", re.IGNORECASE),
    re.compile(r"/(?:p|item|dp)/[\w-]+/?$", re.IGNORECASE),
]

_TEXT_TAGS = ("h1", "h2", "h3", "p", "li", "span", "div")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9]+(?:[-_/][A-Za-zА-Яа-я0-9]+)*")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _token_count(text: str) -> int:
    return len(_tokenize(text))


def _trim_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    toks = _tokenize(text)
    return " ".join(toks[:max_tokens])


def _is_price(text: str) -> bool:
    text = text.strip()
    return bool(
        re.match(
            r"^[\$€£¥₹]?\s?\d[\d,.\s]*(?:[\$€£¥₹]|usd|eur)?$",
            text,
            re.I,
        )
    )


def _is_noise(text: str) -> bool:
    text = _clean(text)
    if not text:
        return True
    if len(text) < 3:
        return True
    if _BAD_TEXT_RE.search(text):
        return True
    if _is_price(text):
        return True
    if re.fullmatch(r"[\W_]+", text):
        return True
    if sum(ch.isalpha() for ch in text) < 2:
        return True
    return False


def _in_skip_zone(tag: Tag) -> bool:
    for parent in [tag, *tag.parents]:
        if not isinstance(parent, Tag):
            continue
        if parent.name in _SKIP_TAGS:
            return True
        cls = " ".join(parent.get("class", []))
        id_ = parent.get("id", "")
        role = parent.get("role", "")
        check = f"{cls} {id_} {role}"
        if _SKIP_ATTRS.search(check):
            return True
    return False


def _fetch(url: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if response.status_code == 200:
            return BeautifulSoup(response.text, "html.parser"), response.url, response.status_code
        return None, response.url, response.status_code
    except Exception:
        return None, url, 0


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        key = x.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _walk_ld(node, out: list[str]):
    if isinstance(node, list):
        for item in node:
            _walk_ld(item, out)
        return

    if not isinstance(node, dict):
        return

    types = node.get("@type", "")
    types = types if isinstance(types, list) else [types]

    if any(t in ("Product", "IndividualProduct") for t in types):
        name = _clean(node.get("name", ""))
        if name and not _is_noise(name):
            out.append(name)
        return

    if "ItemList" in types:
        for elem in node.get("itemListElement", []):
            _walk_ld(elem, out)

    if "@graph" in node:
        _walk_ld(node["@graph"], out)

    for key, val in node.items():
        if key.startswith("@"):
            continue
        if isinstance(val, (dict, list)):
            _walk_ld(val, out)


def _from_jsonld(soup: BeautifulSoup) -> list[str]:
    products = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        _walk_ld(data, products)
    return products


def _looks_like_product_url(href: str, base_host: str) -> bool:
    parsed = urlparse(href)
    if parsed.netloc and parsed.netloc != base_host:
        return False
    return any(pat.search(parsed.path) for pat in _PRODUCT_URL_PATTERNS)


def _name_from_img_link(a_tag: Tag):
    img = a_tag.find("img")
    if not img:
        return None
    for attr in ("title", "alt"):
        val = _clean(img.get(attr, "") or "")
        if val and not _is_noise(val) and not _is_price(val):
            return val
    return None


def _name_from_text_link(a_tag: Tag):
    lines = [_clean(x) for x in a_tag.get_text("\n").split("\n") if _clean(x)]
    for line in lines:
        if not _is_noise(line) and not _is_price(line):
            return line
    return None


def _from_product_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    base_host = urlparse(base_url).netloc
    found = {}

    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        norm = urlparse(href)._replace(query="", fragment="").geturl()

        if not _looks_like_product_url(href, base_host):
            continue
        if _in_skip_zone(a):
            continue
        if norm in found:
            continue

        name = _name_from_img_link(a) or _name_from_text_link(a)
        if name:
            found[norm] = name

    return list(found.values())


def _name_from_card(card: Tag):
    for level in ("h1", "h2", "h3", "h4", "h5"):
        h = card.find(level)
        if h:
            txt = _clean(h.get_text())
            if txt and not _is_noise(txt) and not _is_price(txt):
                return txt

    for el in card.find_all(True):
        cls = " ".join(el.get("class", []))
        if re.search(r"(product|item|card)[_-]?(name|title)", cls, re.I):
            txt = _clean(el.get_text())
            if txt and not _is_noise(txt) and not _is_price(txt):
                return txt

    a = card.find("a")
    if a:
        return _name_from_text_link(a)

    return None


def _from_html_cards(soup: BeautifulSoup) -> list[str]:
    products = []
    seen = set()

    candidates = [
        tag for tag in soup.find_all(True)
        if _CARD_PATTERN.search(" ".join(tag.get("class", [])))
    ]

    filtered = []
    for c in candidates:
        if not any(c in f.descendants for f in filtered):
            filtered = [f for f in filtered if f not in c.descendants]
            filtered.append(c)

    for card in filtered:
        if _in_skip_zone(card):
            continue
        name = _name_from_card(card)
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            products.append(name)

    return products


def _from_title(soup: BeautifulSoup):
    title_tag = soup.find("title")
    if title_tag:
        raw = _clean(title_tag.get_text())
        txt = re.split(r"\s*[|–—]\s*", raw)[0].strip()
        txt = re.sub(r"\s*[-–]\s*(Shop|Store|Buy|Online).*$", "", txt, flags=re.I).strip()
        if txt and not _is_noise(txt):
            return txt
    return None


def _detect_page_type(products: list[str]) -> str:
    return "catalog" if len(products) >= 6 else "single"


def _collect_noise_candidates(soup: BeautifulSoup, products: list[str], page_type: str) -> list[str]:
    product_keys = {p.casefold() for p in products}
    seen = set()
    out = []

    for tag in soup.find_all(_TEXT_TAGS):
        if _in_skip_zone(tag):
            continue
        if tag.find(_TEXT_TAGS):
            continue

        txt = _clean(tag.get_text(" ", strip=True))
        if _is_noise(txt):
            continue
        if txt.casefold() in product_keys:
            continue

        n = _token_count(txt)
        max_piece = 12 if page_type == "single" else 16

        if n > max_piece:
            txt = _trim_to_tokens(txt, max_piece)
            n = _token_count(txt)

        if n < 2:
            continue

        key = txt.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(txt)

    return out


def _take_noise_by_budget(candidates: list[str], max_tokens: int) -> list[str]:
    picked = []
    cur = 0

    for txt in candidates:
        if cur >= max_tokens:
            break

        n = _token_count(txt)
        if n <= 0:
            continue

        if cur + n <= max_tokens:
            picked.append(txt)
            cur += n
        else:
            remain = max_tokens - cur
            clipped = _trim_to_tokens(txt, remain)
            if clipped:
                picked.append(clipped)
            break

    return picked


def _stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def scrape_some_page_info(
    url: str,
    include_status: bool = False,
    shuffle_blocks: bool = False,
) -> str:
    soup, final_url, status_code = _fetch(url)
    if not soup:
        return f"status_{status_code}" if include_status else ""

    products = _dedupe_keep_order(
        _from_jsonld(soup) +
        _from_html_cards(soup) +
        _from_product_links(soup, final_url)
    )

    title = _from_title(soup)
    if title and title.casefold() not in {p.casefold() for p in products}:
        products = [title] + products

    page_type = _detect_page_type(products) if products else "single"
    noise_candidates = _collect_noise_candidates(soup, products, page_type)

    noise_budget = 120 if products else 80
    noise = _take_noise_by_budget(noise_candidates, max_tokens=noise_budget)

    blocks = []

    if include_status:
        blocks.append(f"status_{status_code}")

    if products:
        blocks.extend(products[:40])
    blocks.extend(noise)

    blocks = _dedupe_keep_order([_clean(x) for x in blocks if _clean(x)])

    if shuffle_blocks:
        rnd = random.Random(_stable_seed(url))
        rnd.shuffle(blocks)

    return "\n".join(blocks)