from bs4 import BeautifulSoup

with open('employment/age.html', 'r', encoding='utf-8') as f:
    soup = BeautifulSoup(f, 'html.parser')
    text = soup.get_text()
    words = text.split()
    print(f"Word count: {len(words)}")