#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to download and prepare training datasets for RNNMorph.

Downloads all publicly available training data from morphoRuEval-2017
and converts it to the format required by RNNMorph.

Usage:
    python download_training_data.py

Data will be downloaded to: rnnmorph/datasets/
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path

# Configuration
DATASETS_DIR = Path(__file__).parent / "rnnmorph" / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PREPARED_DIR = DATASETS_DIR / "prepared"

# Create directories
RAW_DIR.mkdir(parents=True, exist_ok=True)
PREPARED_DIR.mkdir(parents=True, exist_ok=True)


def print_header(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70 + "\n")


def run_command(cmd, description):
    """Run a shell command with progress indication."""
    print(f"[INFO] {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] Command failed: {cmd}")
        print(f"       stderr: {result.stderr[:200]}")
        return False
    return True


def download_file(url, output_path, description, headers=None):
    """Download a file with progress and optional custom headers."""
    print(f"[INFO] Downloading {description}...")
    print(f"       URL: {url}")
    print(f"       Path: {output_path}")

    if output_path.exists():
        print(f"[INFO] File already exists, skipping...")
        return True

    try:
        # Use urllib with optional headers for browser emulation
        req = urllib.request.Request(url)
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        with urllib.request.urlopen(req, timeout=300) as response:
            with open(output_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        print(f"[INFO] Download complete: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract zip/tar/gz archives."""
    print(f"[INFO] Extracting {archive_path.name}...")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz':
            output_path = extract_to / archive_path.stem
            with gzip.open(archive_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif archive_path.suffix in ['.tar', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"[WARN] Unknown archive format: {archive_path.suffix}")
            return False
        
        print(f"[INFO] Extraction complete")
        return True
    except Exception as e:
        print(f"[WARN] Extraction failed: {e}")
        return False


###############################################################################
# Dataset 1: morphoRuEval-2017 Training Data
###############################################################################

def download_morphorueval_2017():
    """Download morphoRuEval-2017 training data from GitHub."""
    print_header("Dataset 1: morphoRuEval-2017")
    
    repo_url = "https://github.com/dialogue-evaluation/morphoRuEval-2017.git"
    repo_dir = RAW_DIR / "morphoRuEval-2017"
    
    if repo_dir.exists():
        print(f"[INFO] Repository already exists, pulling latest...")
        run_command(f"cd {repo_dir} && git pull", "Updating morphoRuEval-2017")
    else:
        run_command(f"git clone {repo_url} {repo_dir}", "Cloning morphoRuEval-2017")
    
    # List available data
    print("\n[INFO] Available training data:")
    data_dirs = [
        repo_dir / "training_data",
        repo_dir / "UD_training_data",
        repo_dir / "plain_text",
    ]
    
    for data_dir in data_dirs:
        if data_dir.exists():
            print(f"  - {data_dir.relative_to(repo_dir)}")
            for f in data_dir.iterdir():
                if f.is_file():
                    print(f"    * {f.name}")
    
    return repo_dir


###############################################################################
# Dataset 2: Universal Dependencies Russian Corpora
###############################################################################

def download_ud_russian():
    """Download Universal Dependencies Russian corpora."""
    print_header("Dataset 2: Universal Dependencies Russian")
    
    # UD Russian SynTagRus
    syntagrus_url = "https://github.com/UniversalDependencies/UD_Russian-SynTagRus/archive/refs/heads/master.zip"
    syntagrus_path = RAW_DIR / "UD_Russian-SynTagRus"
    download_file(syntagrus_url, RAW_DIR / "syntagrus.zip", "UD Russian SynTagRus")
    extract_archive(RAW_DIR / "syntagrus.zip", RAW_DIR)
    
    # UD Russian GSD
    gsd_url = "https://github.com/UniversalDependencies/UD_Russian-GSD/archive/refs/heads/master.zip"
    gsd_path = RAW_DIR / "UD_Russian-GSD"
    download_file(gsd_url, RAW_DIR / "gsd.zip", "UD Russian GSD")
    extract_archive(RAW_DIR / "gsd.zip", RAW_DIR)
    
    # UD Russian Taiga
    taiga_url = "https://github.com/UniversalDependencies/UD_Russian-Taiga/archive/refs/heads/master.zip"
    taiga_path = RAW_DIR / "UD_Russian-Taiga"
    download_file(taiga_url, RAW_DIR / "taiga.zip", "UD Russian Taiga")
    extract_archive(RAW_DIR / "taiga.zip", RAW_DIR)
    
    # UD Russian PUD
    pud_url = "https://github.com/UniversalDependencies/UD_Russian-PUD/archive/refs/heads/master.zip"
    pud_path = RAW_DIR / "UD_Russian-PUD"
    download_file(pud_url, RAW_DIR / "pud.zip", "UD Russian PUD")
    extract_archive(RAW_DIR / "pud.zip", RAW_DIR)
    
    print("\n[INFO] Downloaded UD Russian corpora:")
    for corpus in [syntagrus_path, gsd_path, taiga_path, pud_path]:
        if corpus.exists():
            conllu_files = list(corpus.glob("*.conllu"))
            print(f"  - {corpus.name}: {len(conllu_files)} files")
    
    return [syntagrus_path, gsd_path, taiga_path, pud_path]


###############################################################################
# Dataset 3: OpenCorpora Full Annotated Corpus
###############################################################################

def download_opencorpora():
    """Download OpenCorpora annotated corpus."""
    print_header("Dataset 3: OpenCorpora")

    # Browser headers to avoid 403 Forbidden
    browser_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }

    # OpenCorpora full annotated corpus with resolved homonymy
    # Correct URL from https://opencorpora.org/?page=downloads
    opencorpora_url = "https://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip"
    output_path = RAW_DIR / "annot.opcorpora.xml.zip"

    # Try primary URL with browser headers
    if not download_file(opencorpora_url, output_path, "OpenCorpora Annotated Corpus (XML ZIP)", headers=browser_headers):
        # Fallback to HTTP
        opencorpora_http_url = "http://opencorpora.org/files/export/annot/annot.opcorpora.xml.zip"
        if not download_file(opencorpora_http_url, output_path, "OpenCorpora Annotated Corpus (HTTP)", headers=browser_headers):
            # Fallback to bz2 format
            opencorpora_bz2_url = "http://opencorpora.org/files/export/annot/annot.opcorpora.xml.bz2"
            output_path = RAW_DIR / "annot.opcorpora.xml.bz2"
            download_file(opencorpora_bz2_url, output_path, "OpenCorpora Annotated Corpus (BZ2)", headers=browser_headers)

    extract_archive(output_path, RAW_DIR)

    # Also download dictionary (correct URL from opencorpora.org)
    dict_url = "https://opencorpora.org/files/export/dict/dict.opencorpora.xml.zip"
    dict_path = RAW_DIR / "dict.opencorpora.xml.zip"
    download_file(dict_url, dict_path, "OpenCorpora Dictionary", headers=browser_headers)
    extract_archive(dict_path, RAW_DIR)

    print("\n[INFO] OpenCorpora files downloaded:")
    xml_file = RAW_DIR / "annot.opcorpora.xml"
    if xml_file.exists():
        size_mb = xml_file.stat().st_size / 1024 / 1024
        print(f"  - annot.opcorpora.xml: {size_mb:.2f} MB")
    
    return RAW_DIR


###############################################################################
# Dataset 4: Russian National Corpus (RNC)
###############################################################################

def download_rnc():
    """
    Download Russian National Corpus disambiguated subcorpus.
    
    Sources:
    - morphoRuEval-2017 RNC texts (UD format)
    - SynTagRus (RNC-based, already downloaded via UD)
    """
    print_header("Dataset 4: Russian National Corpus (RNC)")
    
    # RNC from morphoRuEval-2017 (UD format, manually disambiguated)
    rnc_url = "https://github.com/dialogue-evaluation/morphoRuEval-2017/raw/master/RNC_texts.rar"
    output_path = RAW_DIR / "RNC_texts.rar"
    download_file(rnc_url, output_path, "RNC UD texts (morphoRuEval)")
    
    # Try to extract RAR file
    if output_path.exists():
        extracted = False
        
        # Try unar first (macOS, works with RAR)
        try:
            if run_command(f"unar -o {RAW_DIR} -f {output_path}", "Extracting RNC texts with unar"):
                extracted = True
                print("[INFO] Extracted with unar")
        except Exception as e:
            pass
        
        # Try unrar (Linux)
        if not extracted:
            try:
                if run_command(f"unrar x -o+ {output_path} {RAW_DIR}", "Extracting RNC texts with unrar"):
                    extracted = True
                    print("[INFO] Extracted with unrar")
            except Exception as e:
                pass
        
        # Last resort: Python library
        if not extracted:
            print("[WARN] Neither unar nor unrar available")
            print("       Install: brew install unar (macOS) or sudo apt-get install unrar (Linux)")
            print("       Or use Python library: pip install libarchive-c")
            try:
                import libarchive
                libarchive.extract_file(str(output_path), str(RAW_DIR))
                print("[INFO] Extracted with libarchive")
                extracted = True
            except:
                print("[INFO] Alternative: RNC data is also available in UD SynTagRus (already downloaded)")
    
    # Alternative: RNC Poetry UD
    poetry_url = "https://github.com/UniversalDependencies/UD_Russian-Poetry/archive/refs/heads/master.zip"
    poetry_path = RAW_DIR / "UD_Russian-Poetry.zip"
    download_file(poetry_url, poetry_path, "RNC Poetry UD")
    extract_archive(poetry_path, RAW_DIR)
    
    print("\n[INFO] RNC files downloaded:")
    rnc_file = RAW_DIR / "RNC_texts"
    if rnc_file.exists():
        for f in rnc_file.glob("*.txt"):
            print(f"  - {f.name}: {f.stat().st_size / 1024 / 1024:.2f} MB")
    
    return RAW_DIR


###############################################################################
# Dataset 5: GIKRYA (Internet Corpus) Full
###############################################################################

def download_gikrya_full():
    """
    Download full GIKRYA corpus from morphoRuEval-2017.
    """
    print_header("Dataset 5: GIKRYA Full (Internet Corpus)")
    
    # GIKRYA full texts from morphoRuEval
    gikrya_url = "https://github.com/dialogue-evaluation/morphoRuEval-2017/raw/master/GIKRYA_texts_new.zip"
    output_path = RAW_DIR / "GIKRYA_texts.zip"
    download_file(gikrya_url, output_path, "GIKRYA full texts")
    extract_archive(output_path, RAW_DIR)
    
    print("\n[INFO] GIKRYA files downloaded:")
    # Check for both possible extraction results
    gikrya_dir = RAW_DIR / "GIKRYA_texts"
    if gikrya_dir.exists():
        for f in gikrya_dir.glob("*.txt"):
            print(f"  - {f.name}: {f.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Also check for .out files (alternative extraction)
    for f in RAW_DIR.glob("gikrya_new_*.out"):
        print(f"  - {f.name}: {f.stat().st_size / 1024 / 1024:.2f} MB")
    
    return RAW_DIR


###############################################################################
# Conversion Functions
###############################################################################

def convert_conllu_to_rnnmorph(conllu_path, output_path):
    """
    Convert CoNLL-U format to RNNMorph tab-separated format.
    
    CoNLL-U format:
    ID    FORM    LEMMA    UPOS    XPOS    FEATS    HEAD    DEPREL    DEPS    MISC
    
    RNNMorph format:
    FORM    LEMMA    UPOS    FEATS
    """
    print(f"[INFO] Converting {conllu_path.name}...")
    
    sentences = 0
    words = 0
    
    with open(conllu_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                if not line and sentences > 0:
                    f_out.write('\n')  # Sentence separator
                continue
            
            parts = line.split('\t')
            if len(parts) < 8:
                continue
            
            # Skip multiword tokens (e.g., "1-2")
            if '-' in parts[0]:
                continue
            
            try:
                form = parts[1]
                lemma = parts[2]
                upos = parts[3]
                feats = parts[5]
                
                # Replace underscore with empty for features
                if feats == '_':
                    feats = '_'
                
                f_out.write(f"{form}\t{lemma}\t{upos}\t{feats}\n")
                words += 1
                
            except (IndexError, ValueError) as e:
                continue
        
        sentences = words // 15  # Rough estimate
    
    print(f"[INFO] Converted: {words:,} words (est. {sentences:,} sentences)")
    return words


def prepare_ud_corpus(ud_path, output_dir):
    """Prepare a UD corpus for training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_words = 0
    output_file = output_dir / f"{ud_path.name}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for conllu_file in ud_path.glob("*.conllu"):
            words = convert_conllu_to_rnnmorph(conllu_file, output_file)
            total_words += words
    
    print(f"[INFO] Total for {ud_path.name}: {total_words:,} words")
    return total_words


###############################################################################
# Main Preparation Pipeline
###############################################################################

def convert_opencorpora_xml_to_rnnmorph(xml_path, output_path):
    """
    Convert OpenCorpora XML format to RNNMorph tab-separated format.
    
    OpenCorpora XML structure (new format):
    <token id="1" text="Школа">
      <tfr rev_id="834910" t="Школа">
        <v>
          <l id="380220" t="школа">
            <g v="NOUN"/><g v="inan"/><g v="femn"/>...
    
    RNNMorph format:
    word<TAB>lemma<TAB>POS<TAB>grammemes
    """
    print(f"[INFO] Converting OpenCorpora XML {xml_path.name}...")
    
    import xml.etree.ElementTree as ET
    
    words = 0
    sentences = 0
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Find all tokens in the XML
            for token_elem in root.iter('token'):
                try:
                    # Get word form from token text attribute
                    form = token_elem.get('text', '').strip()
                    if not form:
                        continue
                    
                    # Find lemma and grammemes in tfr/v/l/g elements
                    lemma = ''
                    pos = ''
                    grammemes_list = []
                    
                    tfr_elem = token_elem.find('tfr')
                    if tfr_elem is not None:
                        v_elem = tfr_elem.find('v')
                        if v_elem is not None:
                            l_elem = v_elem.find('l')
                            if l_elem is not None:
                                lemma = l_elem.get('t', '')
                                
                                # Get all grammemes from <g> elements
                                g_elems = l_elem.findall('g')
                                if g_elems:
                                    pos = g_elems[0].get('v', '')
                                    # Get all other grammemes
                                    for g_elem in g_elems[1:]:
                                        grammemes_list.append(g_elem.get('v', ''))
                    
                    grammemes = '|'.join(grammemes_list) if grammemes_list else '_'
                    
                    if lemma and pos:
                        f_out.write(f"{form}\t{lemma}\t{pos}\t{grammemes}\n")
                        words += 1
                    
                    # Check for sentence boundary
                    if token_elem.get('text', '').strip() in ['.', '!', '?', '…']:
                        f_out.write('\n')
                        sentences += 1
                        
                except Exception as e:
                    continue
        
        print(f"[INFO] Converted: {words:,} words, {sentences:,} sentences")
        return words
    except Exception as e:
        print(f"[WARN] XML conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def prepare_all_datasets():
    """Prepare all downloaded datasets for training."""
    print_header("Preparing Datasets")
    
    prepared_files = []
    
    # 1. Prepare UD Russian corpora
    print("\n[STEP 1] Preparing UD Russian corpora...")
    ud_corpora = [
        RAW_DIR / "UD_Russian-SynTagRus-master",
        RAW_DIR / "UD_Russian-GSD-master",
        RAW_DIR / "UD_Russian-Taiga-master",
        RAW_DIR / "UD_Russian-PUD-master",
    ]
    
    combined_ud = PREPARED_DIR / "ud_combined.txt"
    total_ud_words = 0
    
    with open(combined_ud, 'w', encoding='utf-8') as f_out:
        for corpus in ud_corpora:
            if corpus.exists():
                print(f"  Processing {corpus.name}...")
                temp_file = PREPARED_DIR / f"{corpus.name}.txt"
                words = prepare_ud_corpus(corpus, PREPARED_DIR)
                total_ud_words += words
                
                # Append to combined
                with open(temp_file, 'r', encoding='utf-8') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                f_out.write('\n')
    
    prepared_files.append(('UD Combined', combined_ud, total_ud_words))

    # 2. Prepare OpenCorpora
    print("\n[STEP 2] Preparing OpenCorpora...")
    opencorpora_xml = RAW_DIR / "annot.opcorpora.xml"
    if opencorpora_xml.exists():
        opencorpora_output = PREPARED_DIR / "opencorpora_annotated.txt"
        words = convert_opencorpora_xml_to_rnnmorph(opencorpora_xml, opencorpora_output)
        if words > 0:
            prepared_files.append(('OpenCorpora', opencorpora_output, words))
            print(f"[INFO] OpenCorpora prepared: {words:,} words")
    else:
        print("[INFO] OpenCorpora XML not found, skipping...")
        print("       Download manually from: https://opencorpora.org/?page=downloads")
        print(f"       Place file in: {RAW_DIR}/annot.opcorpora.xml.zip")

    # 3. Prepare RNC
    print("\n[STEP 3] Preparing RNC...")
    rnc_dir = RAW_DIR / "RNC_texts"
    rnc_files = []
    
    # Check for RNC files in various formats
    if rnc_dir.exists():
        rnc_files = list(rnc_dir.glob("*.txt"))
    
    # Also check for CoNLL format from unar extraction
    rnc_conll = RAW_DIR / "RNCgoldInUD_Morpho.conll"
    if rnc_conll.exists():
        rnc_files.append(rnc_conll)
    
    if rnc_files:
        rnc_output = PREPARED_DIR / "rnc_texts.txt"
        words = 0
        with open(rnc_output, 'w', encoding='utf-8') as f_out:
            for txt_file in rnc_files:
                print(f"  Processing {txt_file.name}...")
                
                # Handle CoNLL format
                if txt_file.suffix == '.conll':
                    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line or line.startswith('=='):
                                f_out.write('\n')
                                continue
                            if line.startswith('#') or line.startswith('==>'):
                                continue
                            parts = line.split('\t')
                            if len(parts) >= 4:
                                # Format: empty, word, lemma, POS, grammemes, extra
                                word = parts[1].strip()
                                lemma = parts[2].strip()
                                pos = parts[3].strip()
                                grammemes = parts[4].strip() if len(parts) > 4 else '_'
                                if word and lemma and pos:
                                    f_out.write(f"{word}\t{lemma}\t{pos}\t{grammemes}\n")
                                    words += 1
                else:
                    # Handle regular text format
                    with open(txt_file, 'r', encoding='utf-8') as f_in:
                        shutil.copyfileobj(f_in, f_out)
                    f_out.write('\n')
                    with open(txt_file, 'r', encoding='utf-8') as f_in:
                        words += sum(1 for line in f_in if line.strip() and not line.startswith('#'))
        
        prepared_files.append(('RNC', rnc_output, words))
        print(f"[INFO] RNC prepared: {words:,} words")
    else:
        print("[INFO] RNC texts not found, skipping...")

    # 4. Prepare GIKRYA full
    print("\n[STEP 4] Preparing GIKRYA full texts...")
    gikrya_dir = RAW_DIR / "GIKRYA_texts"
    gikrya_files = []
    
    # Check for .txt files in directory
    if gikrya_dir.exists():
        gikrya_files = list(gikrya_dir.glob("*.txt"))
    
    # Also check for .out files (alternative format)
    gikrya_files.extend(list(RAW_DIR.glob("gikrya_new_*.out")))
    
    if gikrya_files:
        gikrya_output = PREPARED_DIR / "gikrya_texts.txt"
        words = 0
        with open(gikrya_output, 'w', encoding='utf-8') as f_out:
            for txt_file in gikrya_files:
                print(f"  Processing {txt_file.name}...")
                # Convert format if needed (remove first column with word index)
                with open(txt_file, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        line = line.strip()
                        if not line:
                            f_out.write('\n')
                            continue
                        if line.startswith('#'):
                            continue
                        parts = line.split('\t')
                        if len(parts) >= 4:
                            # Remove word index (first column)
                            f_out.write(f"{parts[1]}\t{parts[2]}\t{parts[3]}\t{parts[4] if len(parts) > 4 else '_'}\n")
                            words += 1
                        else:
                            f_out.write(f"{line}\n")
                            words += 1
                    f_out.write('\n')
        prepared_files.append(('GIKRYA', gikrya_output, words))
        print(f"[INFO] GIKRYA prepared: {words:,} words")
    else:
        print("[INFO] GIKRYA texts not found, skipping...")

    # 5. Prepare morphoRuEval-2017 data
    print("\n[STEP 5] Preparing morphoRuEval-2017 data...")
    morpho_dir = RAW_DIR / "morphoRuEval-2017" / "UD_training_data"
    if morpho_dir.exists():
        morpho_output = PREPARED_DIR / "morphorueval_2017.txt"
        words = 0
        with open(morpho_output, 'w', encoding='utf-8') as f_out:
            for data_file in morpho_dir.glob("*.txt"):
                print(f"  Copying {data_file.name}...")
                with open(data_file, 'r', encoding='utf-8') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                f_out.write('\n')
                # Count words
                with open(data_file, 'r', encoding='utf-8') as f_in:
                    words += sum(1 for line in f_in if line.strip() and not line.startswith('#'))
        
        prepared_files.append(('morphoRuEval-2017', morpho_output, words))
    
    # 3. Create combined training file
    print("\n[STEP 3] Creating combined training file...")
    combined_all = PREPARED_DIR / "training_combined.txt"
    total_words = 0
    
    with open(combined_all, 'w', encoding='utf-8') as f_out:
        for name, file_path, words in prepared_files:
            if file_path.exists():
                print(f"  Adding {name} ({words:,} words)...")
                with open(file_path, 'r', encoding='utf-8') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                f_out.write('\n')
                total_words += words
    
    prepared_files.append(('COMBINED', combined_all, total_words))
    
    # Summary
    print_header("Preparation Summary")
    print(f"\n{'Dataset':<25} {'Words':>15} {'File':<50}")
    print("-" * 90)
    for name, file_path, words in prepared_files:
        size_mb = file_path.stat().st_size / 1024 / 1024 if file_path.exists() else 0
        print(f"{name:<25} {words:>15,} {file_path.name:<50} ({size_mb:.1f} MB)")
    
    print("-" * 90)
    print(f"{'TOTAL':<25} {total_words:>15,}")
    
    return combined_all


###############################################################################
# Alternative: Create Sample Dataset for Testing
###############################################################################

def create_sample_dataset():
    """
    Create a small sample dataset for testing the training pipeline.
    Uses sentences from public domain Russian literature.
    """
    print_header("Creating Sample Dataset (for testing)")
    
    sample_sentences = """мама	мама	NOUN	Case=Nom|Gender=Fem|Number=Sing
мыла	мыть	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act
раму	рама	NOUN	Case=Acc|Gender=Fem|Number=Sing

кот	кот	NOUN	Case=Nom|Gender=Masc|Number=Sing
сидит	сидеть	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act
на	на	ADP	_
окне	окно	NOUN	Case=Loc|Gender=Neut|Number=Sing

привет	привет	NOUN	Case=Nom|Gender=Masc|Number=Sing
как	как	CONJ	_
дела	дело	NOUN	Case=Nom|Gender=Neut|Number=Plur

я	я	PRON	Case=Nom|Number=Sing|Person=1
люблю	любить	VERB	Mood=Ind|Number=Sing|Person=1|Tense=Notpast|VerbForm=Fin|Voice=Act
программировать	программировать	VERB	VerbForm=Inf

сегодня	сегодня	ADV	Degree=Pos
хороший	хороший	ADJ	Degree=Pos|Case=Nom|Gender=Masc|Number=Sing
день	день	NOUN	Case=Nom|Gender=Masc|Number=Sing

он	он	PRON	Case=Nom|Gender=Masc|Number=Sing|Person=3
читает	читать	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act
книгу	книга	NOUN	Case=Acc|Gender=Fem|Number=Sing

она	она	PRON	Case=Nom|Gender=Fem|Number=Sing|Person=3
пишет	писать	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act
письмо	письмо	NOUN	Case=Acc|Gender=Neut|Number=Sing

мы	мы	PRON	Case=Nom|Number=Plur|Person=1
идем	идти	VERB	Mood=Ind|Number=Plur|Person=1|Tense=Notpast|VerbForm=Fin|Voice=Act
домой	домой	ADV	Degree=Pos

они	они	PRON	Case=Nom|Number=Plur|Person=3
работают	работать	VERB	Mood=Ind|Number=Plur|Person=3|Tense=Notpast|VerbForm=Fin|Voice=Act
в	в	ADP	_
офисе	офис	NOUN	Case=Loc|Gender=Masc|Number=Sing

москва	москва	PROPN	Case=Nom|Gender=Fem|Number=Sing
столица	столица	NOUN	Case=Nom|Gender=Fem|Number=Sing
россии	россия	PROPN	Case=Gen|Gender=Fem|Number=Sing

санкт	санкт	PROPN	Case=Nom|Gender=Masc|Number=Sing
петербург	петербург	PROPN	Case=Nom|Gender=Masc|Number=Sing
культурная	культурный	ADJ	Degree=Pos|Case=Nom|Gender=Fem|Number=Sing
столица	столица	NOUN	Case=Nom|Gender=Fem|Number=Sing

русский	русский	ADJ	Degree=Pos|Case=Nom|Gender=Masc|Number=Sing
язык	язык	NOUN	Case=Nom|Gender=Masc|Number=Sing
очень	очень	ADV	Degree=Pos
богатый	богатый	ADJ	Degree=Pos|Case=Nom|Gender=Masc|Number=Sing
и	и	CONJ	_
красивый	красивый	ADJ	Degree=Pos|Case=Nom|Gender=Masc|Number=Sing

пушкин	пушкин	PROPN	Case=Nom|Gender=Masc|Number=Sing
великий	великий	ADJ	Degree=Pos|Case=Nom|Gender=Masc|Number=Sing
поэт	поэт	NOUN	Case=Nom|Gender=Masc|Number=Sing

толстой	толстой	PROPN	Case=Nom|Gender=Masc|Number=Sing
написал	написать	VERB	Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act
война	война	NOUN	Case=Acc|Gender=Fem|Number=Sing
и	и	CONJ	_
мир	мир	NOUN	Case=Acc|Gender=Masc|Number=Sing

достоевский	достоевский	PROPN	Case=Nom|Gender=Masc|Number=Sing
автор	автор	NOUN	Case=Nom|Gender=Masc|Number=Sing
преступление	преступление	NOUN	Case=Nom|Gender=Neut|Number=Sing
и	и	CONJ	_
наказание	наказание	NOUN	Case=Nom|Gender=Neut|Number=Sing

чехов	чехов	PROPN	Case=Nom|Gender=Masc|Number=Sing
писал	писать	VERB	Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act
рассказы	рассказ	NOUN	Case=Acc|Gender=Masc|Number=Plur
и	и	CONJ	_
пьесы	пьеса	NOUN	Case=Acc|Gender=Fem|Number=Plur

есть	есть	VERB	VerbForm=Inf
много	много	ADV	Degree=Pos
интересный	интересный	ADJ	Degree=Pos|Case=Gen|Gender=Masc|Number=Sing
книга	книга	NOUN	Case=Gen|Gender=Fem|Number=Plur

хотеть	хотеть	VERB	VerbForm=Inf
учиться	учиться	VERB	VerbForm=Inf

мочь	мочь	VERB	VerbForm=Inf
говорить	говорить	VERB	VerbForm=Inf
по	по	ADP	_
русский	русский	ADJ	Degree=Pos|Case=Dat|Gender=Masc|Number=Sing

знать	знать	VERB	VerbForm=Inf
что	что	CONJ	_
делать	делать	VERB	VerbForm=Inf

понимать	понимать	VERB	VerbForm=Inf
как	как	ADV	Degree=Pos
это	это	PRON	Case=Nom|Gender=Neut|Number=Sing
важный	важный	ADJ	Degree=Pos|Case=Nom|Gender=Masc|Number=Sing

думать	думать	VERB	VerbForm=Inf
что	что	CONJ	_
все	всё	PRON	Case=Nom|Gender=Neut|Number=Sing
будет	быть	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Fut|VerbForm=Fin|Voice=Act
хорошо	хорошо	ADV	Degree=Pos
"""
    
    output_path = PREPARED_DIR / "sample_training.txt"
    output_path.write_text(sample_sentences, encoding='utf-8')
    
    word_count = sum(1 for line in sample_sentences.split('\n') if line.strip())
    print(f"[INFO] Created sample dataset: {word_count:,} words")
    print(f"[INFO] Location: {output_path}")
    
    return output_path


###############################################################################
# Main Entry Point
###############################################################################

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and prepare training datasets for RNNMorph"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Download only sample dataset (for testing)"
    )
    parser.add_argument(
        "--morpho", action="store_true",
        help="Download morphoRuEval-2017 data"
    )
    parser.add_argument(
        "--ud", action="store_true",
        help="Download UD Russian corpora"
    )
    parser.add_argument(
        "--opencorpora", action="store_true",
        help="Download OpenCorpora annotated corpus"
    )
    parser.add_argument(
        "--rnc", action="store_true",
        help="Download Russian National Corpus"
    )
    parser.add_argument(
        "--gikrya", action="store_true",
        help="Download GIKRYA full texts"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all available datasets"
    )
    parser.add_argument(
        "--no-prepare", action="store_true",
        help="Skip preparation step"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    # Override output directory if specified
    if args.output_dir:
        global DATASETS_DIR, RAW_DIR, PREPARED_DIR
        DATASETS_DIR = Path(args.output_dir)
        RAW_DIR = DATASETS_DIR / "raw"
        PREPARED_DIR = DATASETS_DIR / "prepared"
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    
    print_header("RNNMorph Training Data Download Script")
    print(f"Data will be downloaded to: {DATASETS_DIR}")
    print(f"Prepared data will be saved to: {PREPARED_DIR}")
    
    # Determine what to download
    download_all = args.all or not any([args.sample, args.morpho, args.ud, args.opencorpora, args.rnc, args.gikrya])
    
    # Download datasets
    if args.morpho or download_all:
        download_morphorueval_2017()
    
    if args.ud or download_all:
        download_ud_russian()
    
    if args.opencorpora or download_all:
        download_opencorpora()
    
    if args.rnc or download_all:
        download_rnc()
    
    if args.gikrya or download_all:
        download_gikrya_full()
    
    if args.sample or download_all:
        create_sample_dataset()
    
    # Prepare datasets
    if not args.no_prepare:
        prepare_all_datasets()
    
    # Final instructions
    print_header("Next Steps")
    print("""
To train the model with the prepared data:

1. Install dependencies:
   pip install keras==2.8.0 tensorflow==2.8.0 pymorphy2 russian-tagsets

2. Run training:
   python -c "
   from rnnmorph.train import train
   train(
       file_names=['rnnmorph/datasets/prepared/training_combined.txt'],
       train_config_path='rnnmorph/models/ru/train_config.json',
       build_config_path='rnnmorph/models/ru/build_config.json',
       language='ru'
   )
   "

3. Or use the Colab notebook for GPU training.

Note: For production-quality models, you need 5-10M+ words of annotated data.
The sample dataset is only for testing the pipeline.
""")
    
    print(f"\nData location: {DATASETS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
