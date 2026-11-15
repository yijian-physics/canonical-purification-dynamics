"""Typeset this report with cached Tectonic; no numerical data are regenerated."""
import re
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent

def escape(text):
    return ''.join({'\\':r'\textbackslash{}','_':r'\_','%':r'\%','#':r'\#','&':r'\&',
                    '{':r'\{','}':r'\}','$':r'\$','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
                    '—':'---','–':'--','×':r'$\times$'}.get(c,c) for c in text)

def inline(text):
    tokens=[]
    def keep(value):
        tokens.append(value);return f'ZZTOKEN{len(tokens)-1}ZZ'
    text=re.sub(r'\$`(.*?)`\$',lambda m:keep('$'+m[1]+'$'),text)
    text=re.sub(r'`([^`]+)`',lambda m:keep(r'\texttt{'+escape(m[1])+'}'),text)
    text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',lambda m:keep(escape(m[1])),text)
    text=escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',lambda m:r'\textbf{'+m[1]+'}',text)
    for i,v in enumerate(tokens):text=text.replace(f'ZZTOKEN{i}ZZ',v)
    return text

lines=(ROOT/'report.md').read_text().splitlines();body=[];i=0
while i<len(lines):
    line=lines[i]
    if line.startswith('```math'):
        block=[];i+=1
        while lines[i]!='```':block.append(lines[i]);i+=1
        body.append('\\[\n'+'\n'.join(block)+'\n\\]')
    elif line.startswith('# '):body.append(r'\maketitle')
    elif line.startswith('### '):
        if 'Answers to the ten' in line: body.append(r'\clearpage')
        body.append(r'\subsection*{'+inline(line[4:])+'}')
    elif line.startswith('## '):body.append(r'\section*{'+inline(line[3:])+'}')
    elif line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[v.strip() for v in lines[i].strip('|').split('|')]
            if not all(re.match(r'^:?-+:?$',c) for c in cells):table.append(cells)
            i+=1
        i-=1;nc=len(table[0]);width=(.94-.025*nc)/nc
        body.append(r'\begingroup\small\setlength{\tabcolsep}{3pt}\begin{longtable}{'+('p{'+str(width)+r'\linewidth}')*nc+'}\\toprule')
        for j,row in enumerate(table):
            body.append(' & '.join(inline(c) for c in row)+r' \\')
            if j==0:body.append(r'\midrule\endhead')
        body.append(r'\bottomrule\end{longtable}\endgroup')
    elif line.startswith('!['):
        m=re.match(r'!\[([^]]+)\]\(([^)]+)\)',line)
        body.append(r'\begin{center}\includegraphics[width=\linewidth,height=.52\textheight,keepaspectratio]{'+m[2]+r'}\end{center}')
    elif re.match(r'^\d+\. ',line):body.append(r'\par\noindent '+inline(line))
    else:body.append(inline(line))
    i+=1
header=r'''\documentclass[10pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{3em}
\title{Matrix-free factor-TDVP\\Force acceleration and remaining accuracy limits}
\author{Canonical purification dynamics research}
\date{November 14, 2026}
\begin{document}
'''
tex=ROOT/'report.tex';tex.write_text(header+'\n'.join(body)+'\n\\end{document}\n')
subprocess.run(['tectonic','--only-cached','--keep-logs','report.tex'],cwd=ROOT,check=True)
