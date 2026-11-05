"""Loopback-only TLS forwarding for Flux when Windows curl resolves the proxy correctly.

Credentials and bodies remain in memory. No request logging, files, open proxy,
TLS bypass, or URL output. The destination is the fixed official Flux service.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess

PROXY = 'socks5h://10.12.201.122:39009'


def quoted(value):
    return '"'+value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r')+'"'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def forward(self):
        if not self.path.startswith('/dev-api/'):
            self.send_error(403); return
        lines = ['url = '+quoted('https://internal.limxdynamics.com'+self.path),
                 'request = '+quoted(self.command)]
        for key in ('X-Api-Key', 'Content-Type', 'Accept'):
            value = self.headers.get(key)
            if value: lines.append('header = '+quoted(key+': '+value))
        size = int(self.headers.get('Content-Length', '0'))
        if size > 1024*1024:
            self.send_error(413); return
        if size:
            lines.append('data-binary = '+quoted(self.rfile.read(size).decode('utf-8')))
        p = subprocess.run(['C:/Windows/System32/curl.exe', '-sS', '--ssl-no-revoke', '--connect-timeout', '15',
                            '--max-time', '90', '--proxy', PROXY, '--config', '-', '-w', '\n%{http_code}'],
                           input=('\n'.join(lines)+'\n').encode(), capture_output=True)
        if p.returncode:
            self.send_response(502); self.end_headers(); self.wfile.write(b'{"error":"official upstream unavailable"}'); return
        body, status = p.stdout.rsplit(b'\n', 1)
        self.send_response(int(status))
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body)

    do_GET = do_POST = do_PUT = do_DELETE = forward


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    print('FLUX_LOCAL_BASE_URL=http://127.0.0.1:'+str(server.server_port)+'/dev-api', flush=True)
    server.serve_forever()
