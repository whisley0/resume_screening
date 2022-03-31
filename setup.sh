mkdir -p ~/.streamlit/
echo "
[general]n
email = "edward_lam@vfc.com"n
" > ~/.streamlit/credentials.toml
echo "
[server]n
headless = truen
enableCORS=falsen
port = $PORTn
" > ~/.streamlit/config.toml
