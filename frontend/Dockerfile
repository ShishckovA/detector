FROM nginx:1.27-alpine

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY frontend/static/index.html /usr/share/nginx/html/index.html
COPY frontend/static/styles.css /usr/share/nginx/html/static/styles.css
COPY frontend/static/app.js /usr/share/nginx/html/static/app.js
COPY frontend/static/assets /usr/share/nginx/html/static/assets

EXPOSE 8080
