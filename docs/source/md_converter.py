import re
import os

def convert_md_to_rst(md_path, rst_path):
    with open(md_path, 'r') as f:
        md_text = f.read()

    lines = md_text.split('\n')
    rst_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # --- Table Handling ---
        if line.strip().startswith('|') and i + 1 < len(lines) and lines[i+1].strip().startswith('|') and '---' in lines[i+1]:
            # Found a table
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            # Parse headers and content
            rows = [line.strip('|').split('|') for line in table_lines if not re.match(r'^\s*\|?[:\s-]*\|', line)]
            if rows:
                headers = [cell.strip() for cell in rows[0]]
                data = [[cell.strip() for cell in row] for row in rows[1:]]
                
                rst_lines.append('.. list-table::')
                rst_lines.append('   :header-rows: 1')
                rst_lines.append('')
                
                # Header row
                rst_lines.append('   * - ' + '\n     - '.join(headers))
                
                # Data rows
                for row in data:
                    # Pad row if it's shorter than headers
                    while len(row) < len(headers):
                        row.append('')
                    rst_lines.append('   * - ' + '\n     - '.join(row))
                rst_lines.append('')
            continue

        # --- Headers ---
        if line.startswith('# '):
            title = line[2:].strip()
            rst_lines.append(title)
            rst_lines.append('=' * len(title))
        elif line.startswith('## '):
            title = line[3:].strip()
            rst_lines.append(title)
            rst_lines.append('-' * len(title))
        elif line.startswith('### '):
            title = line[4:].strip()
            rst_lines.append(title)
            rst_lines.append('~' * len(title))
            
        # --- Code Blocks ---
        elif line.startswith('```'):
            lang = line[3:].strip() or 'text'
            # Clean up known non-Pygments lexers to avoid warnings
            if lang == 'jsonc': lang = 'json'
            if lang == 'jsonl': lang = 'json'
            rst_lines.append(f'.. code-block:: {lang}\n')
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                rst_lines.append('   ' + lines[i]) # Indent code block content
                i += 1
            rst_lines.append('')
            
        # --- Standard Lines ---
        else:
            rst_lines.append(line)
            
        i += 1
            
    with open(rst_path, 'w') as f:
        f.write('\n'.join(rst_lines))
