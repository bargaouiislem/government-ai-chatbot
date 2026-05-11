import base64

with open('logo.png', 'rb') as f:
    logo_b64 = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

with open('index1.html', 'r', encoding='utf-8') as f:
    html = f.read()

html = html.replace('src="logo.png"', f'src="{logo_b64}"')

with open('index1.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('Done! logo.png is now embedded in index1.html')