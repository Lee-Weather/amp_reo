"""Official personal-storage upload, without credentials or signed URLs on disk."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

import flux_ops as flux
from flux_network_bridge import quoted, PROXY

ROOT = Path(__file__).resolve().parents[2]


def request(url, *, method='GET', auth=False, fields=None, file=None):
    if not url.startswith('https://'):
        raise ValueError('Private model transfer requires HTTPS')
    config = ['url = '+quoted(url), 'request = '+quoted(method)]
    if auth: config.append('header = '+quoted('X-Api-Key: '+flux.credentials()['api_key']))
    for key, value in (fields or {}).items():
        config.append('form-string = '+quoted(key+'='+str(value)))
    if file:
        config.append('form = '+quoted('file=@'+file.resolve().as_posix()))
    p = subprocess.run(['C:/Windows/System32/curl.exe', '-sS', '--ssl-no-revoke', '--connect-timeout', '15',
                        '--max-time', '120', '--proxy', PROXY, '--config', '-', '-D', '-', '-w', '\n%{http_code}'],
                       input=('\n'.join(config)+'\n').encode(), capture_output=True)
    if p.returncode: raise RuntimeError('Private input transport failed; request details withheld')
    raw, status = p.stdout.rsplit(b'\n', 1)
    header, body = raw.split(b'\r\n\r\n', 1)
    # curl may include a100 Continue intermediate response for multipart POST.
    while header.startswith(b'HTTP/') and header.splitlines()[0].split()[1] in (b'100', b'103'):
        header, body = body.split(b'\r\n\r\n', 1)
    headers = {}
    for line in header.decode('latin1').splitlines()[1:]:
        if ':' in line:
            key, value = line.split(':', 1); headers[key.lower()] = value.strip()
    return int(status), headers, body


def api(path):
    status, _, body = request('https://internal.limxdynamics.com/dev-api'+path, auth=True)
    if status != 200: raise RuntimeError('Private storage API HTTP '+str(status))
    value = json.loads(body)
    if value.get('code') != 200: raise RuntimeError('Private storage API rejected request')
    return value['data']


def download_location(storage_path):
    status, headers, body = request('https://internal.limxdynamics.com/dev-api/gm/storage/file?'+urlencode({'filePath':storage_path}), auth=True)
    if status in (301, 302, 303, 307, 308):
        location = headers.get('location', '')
        if not location.startswith('https://'): raise ValueError('Private redirect requires HTTPS')
        return location, None
    if status == 200:
        return None, body
    raise RuntimeError('Private input download HTTP '+str(status))


def upload_checkpoint(model, expected_sha256, receipt_path):
    """Use the same private OSS input channel as the official resume dialog."""
    if receipt_path.exists():
        result = json.loads(receipt_path.read_text())
        assert result['sha256'] == expected_sha256 and result['private_acl_verified']
        return result
    assert hashlib.sha256(model.read_bytes()).hexdigest() == expected_sha256
    policy = api('/api/common/upload-directly?'+urlencode({'fileName': 'input_exp115t_teacher_11250.pt'}))
    fields = {'success_action_status':'200', 'policy':policy['policy'], 'x-oss-signature':policy['signature'],
              'x-oss-signature-version':policy['x_oss_signature_version'], 'x-oss-credential':policy['x_oss_credential'],
              'x-oss-date':policy['x_oss_date'], 'key':policy['key'], 'x-oss-security-token':policy['security_token'],
              'x-oss-object-acl':'private'}
    if policy.get('callback'): fields['callback'] = policy['callback']
    status, _, body = request(policy['host'], method='POST', fields=fields, file=model)
    if status != 200:
        # XML contains no credentials, but keep only its public error code.
        import re
        code = re.search(rb'<Code>([^<]+)</Code>', body)
        raise RuntimeError('Private checkpoint upload HTTP '+str(status)+' code='+str(code[1].decode() if code else 'unknown'))
    object_url = policy['host'].rstrip('/')+'/'+policy['key']
    status, _, _ = request(object_url)
    assert status == 403, 'Checkpoint input must reject anonymous reads'
    result = dict(account=flux.ACCOUNT, storage_key=policy['key'], file_name=policy['key'].split('/')[-1],
                  bytes=model.stat().st_size, sha256=expected_sha256, private_acl='private',
                  private_acl_verified=True, official_checkpoint_input=True, signed_urls_persisted=False)
    receipt_path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--account', required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--storage-path', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    flux.ACCOUNT = a.account
    who = flux.call('auth', 'whoami')['data']
    assert who['email'] == a.account
    blob = a.model.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == a.sha256
    if a.receipt.exists(): raise ValueError('Input receipt exists; verify existing object instead of overwriting')
    existing = flux.call('task', 'storage', 'list', '--folder-path', 'personal/')['data']
    assert not any(r.get('name') == a.storage_path.split('/')[-1] for r in existing), 'Destination already exists'
    policy = api('/gm/storage/upload-url?'+urlencode({'fileName':a.storage_path, 'fileSize':len(blob)}))
    fields = {key:value for key,value in policy.items() if key != 'upload_url'}
    fields = {key.replace('_', '-') if key.startswith('x_tos_') else key:value for key,value in fields.items()}
    status, _, _ = request(policy['upload_url'].replace('http://', 'https://'), method='POST', fields=fields, file=a.model)
    if status not in (200, 201, 204): raise RuntimeError('Private upload HTTP '+str(status))
    url, downloaded = download_location(a.storage_path)
    if url:
        code, _, downloaded = request(url)
        assert code == 200
        # A signed object must not be anonymously readable without its query.
        code, _, _ = request(url.split('?', 1)[0])
        assert code in (401, 403), 'Unexpected anonymous read permission; inspect private storage'
    assert hashlib.sha256(downloaded).hexdigest() == a.sha256
    result = dict(account=a.account, user_id=who['userId'], storage_path=a.storage_path,
                  bytes=len(blob), sha256=a.sha256, official_personal_storage=True,
                  download_sha256_verified=True, download_redirect_available=bool(url),
                  signed_urls_persisted=False, verified=True)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
