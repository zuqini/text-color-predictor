def gen_row(background, text):
    return f'''<div class="sample" style="background-color:rgb({background[0] * 255}, {background[1] * 255}, {background[2] * 255});color:{'black' if text[0] == 1 else 'white'};">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. </div>'''

def gen_result_html(result):
    rendered = ''.join(list(map(lambda x: gen_row(x[0], x[1]), result)))
    return f'''<!doctype html>
<html>
<head>
<link href="https://fonts.googleapis.com/css?family=Montserrat" rel="stylesheet" />
<title>Text Color Predictor Results</title>
<style>
body {{
  font-family: 'Montserrat', sans-serif;
  background-color: grey;
}}
</style>
</head>
<body>
<h1>Text Color Predictor Results</h1>
{rendered}
</body>
</html>'''