"""Formatting-only DOCX to official SVPROC LaTeX conversion for TopoFlow.

Run with the managed document Python runtime. The original manuscript is never
modified. All substantive strings are retained in a conversion manifest.
"""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json
import re
import shutil

from docx import Document
from docx.oxml.ns import qn
from lxml import etree as E

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / '.paper_revision/latex_restoration'
DEST = ROOT / 'paper/latex'
SOURCE = ROOT / 'TopoFlow_Atlantis_ICISEA2026.docx'
SOURCE_HASH = 'f9e3c1c178d6845e1e8dc44a3e5b7a97fb872c6fc1ca8d9f5aa29d072d40bf8a'
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(SOURCE.read_bytes()) == SOURCE_HASH, 'Source manuscript changed'
assert (WORK / 'artifact.md').is_file() or (ROOT / 'docs/latex_format_restoration.md').is_file()
WORK.mkdir(parents=True, exist_ok=True)
DEST.mkdir(parents=True, exist_ok=True)
(DEST / 'figures').mkdir(exist_ok=True)
template_hashes = {'svproc.cls':'ddfc1f23edfbb51f13bb6a6e427b63c41cc9a965f08a3bc65e83c81ee9ae9560',
                   'aliascnt.sty':'8d17979fd3ca18f4ac7a379d039f47bf63ffe0409d19c5399d6f3611ed50f9d5',
                   'remreset.sty':'d8b2679ec5c20723a4766690cd46b6bc0ffd20a6555205cefc1521ecb65794a4'}
for name, digest in template_hashes.items():
    candidate = WORK / 'template' / name
    if not candidate.is_file():candidate = DEST / name
    assert sha(candidate.read_bytes()) == digest
    if candidate.resolve() != (DEST / name).resolve():shutil.copyfile(candidate, DEST / name)

M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
greek = {'Φ':r'\Phi','τ':r'\tau','Δ':r'\Delta','α':r'\alpha','μ':r'\mu',
         'π':r'\pi','ρ':r'\rho','σ':r'\sigma','ℓ':r'\ell'}
math_symbols = {**greek,'ŷ':r'\hat{y}','ȳ':r'\bar{y}','∈':r'\in',
                '∪':r'\cup','∑':r'\sum','−':'-','×':r'\times','±':r'\pm'}
subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉ₕₜᵢᵥ','0123456789htiv')
superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵉᵠ','0123456789-eq')

def math_text(value):
    return ''.join(math_symbols.get(c, c) + (' ' if c in math_symbols and c not in ['−'] else '') for c in value)

def math_node(node):
    tag = E.QName(node).localname
    if E.QName(node).namespace != M or tag.endswith('Pr'):
        return ''
    child = lambda name: next((e for e in node if E.QName(e).localname == name), None)
    convert = lambda name: math_node(child(name)) if child(name) is not None else ''
    if tag == 'r':
        value = ''.join(node.xpath('.//m:t/text()', namespaces={'m':M}))
        styles = node.xpath('./m:rPr/m:sty/@m:val', namespaces={'m':M})
        if styles == ['p'] and value in ['exp','log']:
            return '\\' + value + ' '
        if styles == ['p'] and value.strip().isalpha():
            return r'\mathrm{' + value + '}'
        return math_text(value)
    if tag == 'f': return r'\frac{' + convert('num') + '}{' + convert('den') + '}'
    if tag == 'sSup': return '{' + convert('e') + '}^{' + convert('sup') + '}'
    if tag == 'sSub': return '{' + convert('e') + '}_{' + convert('sub') + '}'
    if tag == 'sSubSup': return '{' + convert('e') + '}_{' + convert('sub') + '}^{' + convert('sup') + '}'
    if tag == 'd':
        get = lambda name, default: node.xpath('./m:dPr/m:'+name+'/@m:val',namespaces={'m':M}) or [default]
        begin, end = get('begChr','(')[0], get('endChr',')')[0]
        delim = {'{':r'\{','}':r'\}','|':'|','':' .'}
        return r'\left' + delim.get(begin,begin) + ' ' + convert('e') + r'\right' + delim.get(end,end)
    if tag == 'nary':
        op = (node.xpath('./m:naryPr/m:chr/@m:val',namespaces={'m':M}) or ['∑'])[0]
        lower, upper = convert('sub'), convert('sup')
        hide = node.xpath('./m:naryPr/m:supHide/@m:val',namespaces={'m':M})
        return math_symbols[op] + ('_{'+lower+'}' if lower else '') + ('^{'+upper+'}' if upper and hide != ['1'] else '') + ' ' + convert('e')
    if tag == 't': return math_text(node.text or '')
    if tag in ['oMathPara','oMath','e','num','den','sub','sup']:
        return ''.join(math_node(c) for c in node)
    raise ValueError('Unhandled OMML math node: '+tag)

def text_tex(value, citations=True):
    """Escape source text without paraphrasing, preserving every symbol's meaning."""
    replacements = {}
    def token(tex):
        key = f'ZZPROTECTED{len(replacements)}ZZ'
        replacements[key] = tex
        return key
    def protect_url(match):
        s = match.group(); clean = s.rstrip('.,;')
        return token(r'\url{'+clean+'}') + s[len(clean):]
    value = re.sub(r'https?://[^\s,;]+', protect_url, value)
    if citations:
        value = re.sub(r'\[(\d+)\]',lambda m:token(r'\cite{r'+m.group(1)+'}'),value)
    value = value.replace('log₁₀',token(r'\ensuremath{\log_{10}}'))
    value = value.replace('zɡ',token(r'\ensuremath{z_g}'))
    value = re.sub(r'[₀-₉ₕₜᵢᵥ]+',lambda m:token(r'\textsubscript{'+m.group().translate(subscripts)+'}'),value)
    value = re.sub(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁻ᵉᵠ]+',lambda m:token(r'\ensuremath{^{'+m.group().translate(superscripts)+'}}'),value)
    for c, tex in {**math_symbols, '→':r'\rightarrow','·':r'\cdot'}.items():
        if c in value:value=value.replace(c, token(r'\ensuremath{'+tex+'}'))
    value = value.replace('–','--').replace('’',"'")
    escaped = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
               '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    value = ''.join(escaped.get(c,c) for c in value)
    for key, tex in replacements.items():value=value.replace(key,tex)
    return value

doc = Document(SOURCE)
body = list(doc._element.body)
paragraphs = doc.paragraphs
title = paragraphs[0].text
names = doc.core_properties.author.split('; ')
emails = re.findall(r'[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]+',paragraphs[2].text)
assert names == ['Shaunak Rane','Tanisha Somai','Hemant Maddu','Shivaji D. Pawar']
assert len(emails) == 4
abstract = paragraphs[3].text.removeprefix('Abstract. ')
keywords = paragraphs[4].text.removeprefix('Keywords: ')
manifest = {'source':SOURCE.name,'source_sha256':SOURCE_HASH,'title':title,'authors':names,
            'emails':emails,'abstract':abstract,'keywords':keywords,'blocks':[],
            'template_files':template_hashes}

preamble = r'''% Generated from the retained manuscript without changing research content.
\documentclass[a4paper]{svproc}
\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,ragged2e,etoolbox}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage[section]{placeins}
\usepackage{xurl}
\usepackage[hidelinks,unicode]{hyperref}
\def\UrlFont{\rmfamily}
\renewcommand{\email}[1]{\href{mailto:#1}{{\normalfont #1}}}
\raggedbottom
\emergencystretch=1.5em
\hyphenpenalty=500
\exhyphenpenalty=500
\widowpenalty=10000
\clubpenalty=10000
\setlength{\textfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\intextsep}{12pt plus 2pt minus 2pt}
\setlength{\floatsep}{10pt plus 2pt minus 2pt}
\makeatletter
\patchcmd{\@maketitle}{\vskip .8cm}{\vskip .45cm}{}{\PackageError{topoflow}{SVPROC title spacing hook changed}{}}
\patchcmd{\@maketitle}{\vskip.35cm}{\vskip.2cm}{}{\PackageError{topoflow}{SVPROC author spacing hook changed}{}}
\makeatother
\renewcommand{\topfraction}{.92}
\renewcommand{\bottomfraction}{.88}
\renewcommand{\textfraction}{.06}
\renewcommand{\floatpagefraction}{.82}
\graphicspath{{figures/}}
\begin{document}
\mainmatter
'''
lines = [preamble, r'\title{\protect\Centering '+text_tex(title)+'}',
         r'\titlerunning{Graph Attention for Permeability and Throat Discovery}',
         r'\toctitle{'+text_tex(title)+'}',
         r'\author{Shaunak Rane\thanks{Corresponding author: \email{'+emails[0]+r'}.} \and '+r' \and '.join(names[1:])+'}',
         r'\authorrunning{S. Rane et al.}', r'\tocauthor{'+', '.join(names)+'}',
         r'\institute{'+r'; '.join(r'\email{'+e+'}' for e in emails[1:])+'}',
         r'\maketitle',r'\begin{abstract}',r'\hyphenpenalty=10000\exhyphenpenalty=10000\emergencystretch=3em',
         r'\raggedright '+text_tex(abstract),r'\keywords{'+text_tex(keywords)+'}',r'\end{abstract}']

def table_tex(element, caption, number):
    from docx.table import Table
    table=Table(element,doc._body)
    cells=[[c.text for c in row.cells] for row in table.rows]
    align = r'@{}>{\RaggedRight\arraybackslash}p{.18\linewidth}>{\RaggedRight\arraybackslash}p{.25\linewidth}>{\RaggedRight\arraybackslash}X@{}' if number==1 else (
            r'@{}>{\RaggedRight\arraybackslash}p{.25\linewidth}>{\RaggedRight\arraybackslash}p{.31\linewidth}>{\RaggedRight\arraybackslash}X@{}' if number==2 else
            r'@{}>{\RaggedRight\arraybackslash}p{.30\linewidth}>{\centering\arraybackslash}X>{\centering\arraybackslash}p{.16\linewidth}>{\centering\arraybackslash}p{.16\linewidth}@{}')
    out=[r'\begin{table}[!htbp]',r'\caption{'+text_tex(caption)+'}',r'\label{tab:'+str(number)+'}',
         r'\centering\small',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.18}',
         r'\begin{tabularx}{\linewidth}{'+align+'}',r'\toprule']
    for j,row in enumerate(cells):
        out.append(' & '.join((r'\textbf{'+text_tex(c)+'}' if j==0 else text_tex(c)) for c in row)+r' \\')
        if j==0:out.append(r'\midrule')
    out += [r'\bottomrule',r'\end{tabularx}',r'\end{table}']
    return '\n'.join(out),cells

count_math=count_fig=count_table=0
pending_table=None
refs=[]
for index,element in enumerate(body):
    if element.tag==qn('w:sectPr'):continue
    text=''.join(element.xpath('.//w:t/text()'))
    if element.tag==qn('w:tbl'):
        count_table+=1
        assert pending_table is not None
        tex,cells=table_tex(element,pending_table,count_table)
        lines.append(tex);manifest['blocks'].append({'kind':'table','cells':cells,'caption':pending_table,'latex':tex})
        pending_table=None
        continue
    style=(element.xpath('./w:pPr/w:pStyle/@w:val') or ['Normal'])[0]
    if style in ['Title','author','address','abstract','keywords']:continue
    if style=='tablecaption':
        pending_table=re.sub(r'^Table \d+\.\s*','',text);continue
    if style=='figurecaption':continue
    equations=element.xpath('.//m:oMathPara')
    if equations:
        assert len(equations)==1
        count_math+=1
        tex=math_node(equations[0])
        if lines[-1]=='':lines.pop()
        if index and body[index-1].xpath('.//m:oMathPara'):
            lines.append(r'\nopagebreak[4]')
        lines += [r'\[',tex,r'\]']
        canonical=E.tostring(equations[0],method='c14n',exclusive=True)
        manifest['blocks'].append({'kind':'equation','number':count_math,'omml_sha256':sha(canonical),'latex':tex})
        continue
    images=element.xpath('.//w:drawing')
    if images:
        count_fig+=1
        embed=element.xpath('.//*[@r:embed]')[0].get(qn('r:embed'))
        blob=doc.part.related_parts[embed].blob
        f=DEST/'figures'/f'figure-{count_fig}.png';f.write_bytes(blob)
        caption=''.join(body[index+1].xpath('.//w:t/text()'))
        caption=re.sub(r'^Fig\. \d+\.\s*','',caption)
        tex='\n'.join([r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=.84\linewidth]{figure-'+str(count_fig)+'.png}',
                       r'\caption{'+text_tex(caption)+'}',r'\label{fig:'+str(count_fig)+'}',r'\end{figure}'])
        lines.append(tex);manifest['blocks'].append({'kind':'figure','number':count_fig,'sha256':sha(blob),'caption':caption,'latex':tex})
        continue
    if style=='heading1':
        if text=='References':continue
        lines.append(r'\section{'+text_tex(text)+'}')
    elif style=='heading2':lines.append(r'\subsection{'+text_tex(text)+'}')
    elif style=='referenceitem':
        refs.append(text);continue
    elif text:
        if style=='acknowlegments':
            prefix,content=text.split('. ',1)
            lines.append(r'\paragraph*{'+text_tex(prefix)+'.} '+text_tex(content))
        else:lines.append(text_tex(text))
    else:continue
    manifest['blocks'].append({'kind':'heading' if style in ['heading1','heading2'] else 'paragraph','style':style,'text':text,'latex':lines[-1]})
    lines.append('')

assert count_math==11 and count_fig==4 and count_table==3 and len(refs)==18
lines += [r'\begin{thebibliography}{99}']
for i,reference in enumerate(refs,1):
    tex=text_tex(reference,citations=False)
    lines.append(r'\bibitem{r'+str(i)+'} '+tex)
manifest['references']=refs
lines += [r'\end{thebibliography}',r'\end{document}','']
(DEST/'topoflow.tex').write_text('\n'.join(lines),encoding='utf-8')
(WORK/'content_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
assert sha(SOURCE.read_bytes())==SOURCE_HASH
print(json.dumps({'tex':str(DEST/'topoflow.tex'),'equations':count_math,'figures':count_fig,
                  'tables':count_table,'references':len(refs),'abstract_words':len(abstract.split())},indent=2))
