import urllib.request
import time

replace_dict = {
    '</p>': '\n',
    ' <p>': '',
    '<p>': '',
    '<b>': '',
    '</b>': '',
    '<span>': '',
    '</span>': '',
    '<i>': '',
    '</i>': '',
    '<p />': '',
    '&nbsp;': ' ',
    'xc3xa3': 'ã',
    'xc3xa7': 'ç',
    'xc3xb5': 'õ',
    'xc3xa9': 'é',
    'xc3x89': 'É',
    'xc3xa0': 'à',
    'xc3xba': 'ú',
    'xc3xaa': 'ê',
    'xc3xa1': 'á',
    'xe2x80xa6': '...',
    'xc3xb3': 'ó',
    'xc2xa0': '',
    'xe2x80x94': '',
    'xc3xad': 'í',
    'xc3xb4': 'o',
    'xc3xa2': 'â'


}

f = open("pessoa.txt", "a+")

for nr in range(4, 4545):  # 4 a 4545
    print(nr)
    time.sleep(0.5)
    try:
        response = urllib.request.urlopen('http://arquivopessoa.net/textos/'+str(nr))
        html = str(response.read())
    except urllib.error.HTTPError:
        print("ERROR: number {} does not exist.".format(nr))
        time.sleep(2)


    zone_start = html.find('<div class="texto-poesia">')
    if zone_start != -1:
        zone_start+=28
        print('Poem!')
        poem_mid = html[zone_start:]
        poem_end = poem_mid.find('</div>')
        poem = poem_mid[:poem_end-1]

        poem = poem.replace('\\n', '')
        poem = poem.replace('\\', '')

        for key, value in replace_dict.items():
            poem = poem.replace(key, '{}'.format(value))

        print('\n')
        print(poem)
        f.write(poem+'\n \n')

    else:
        print('Not a poem')
        continue



