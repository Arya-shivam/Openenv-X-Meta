import os

replacements = {
    '\u2014': '-',
    '\u2013': '-',
    '\u201c': '"',
    '\u201d': '"',
    '\u2018': "'",
    '\u2019': "'",
    '\u2500': '-',
    '\u2713': '[PASS]',
    '\U0001f50a': '[Speaker]',
    '\U0001f3e6': '[Bank]'
}

def sanitize_file(path):
    try:
        with open(path, 'rb') as fin:
            data = fin.read()
        
        # Decode the file contents to a string
        text = data.decode('utf-8', 'ignore')
        
        # Apply specifically mapped replacements
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        # Force all other non-ASCII characters to empty strings
        pure_ascii = ''.join(c if ord(c) < 128 else '' for c in text)
        
        # Encode back to pure ASCII as bytes
        with open(path, 'wb') as fout:
            fout.write(pure_ascii.encode('ascii'))
            
        print(f"Sanitized {path}")
    except Exception as e:
        print(f"Failed to sanitize {path}: {e}")

if __name__ == "__main__":
    for root, dirs, files in os.walk('.'):
        # Skip virtual environments and hidden git folders
        if '.venv' in root or '.git' in root or '__pycache__' in root:
            continue
            
        for f in files:
            if f.endswith(('.py', '.yaml', '.toml', '.md', '.txt')):
                sanitize_file(os.path.join(root, f))
