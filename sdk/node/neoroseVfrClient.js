// Minimal Node client for Neorose VFR
// Usage: const client = new VFRClient('http://127.0.0.1:8000');
//        const jobId = await client.tryOn('user.jpg','garment.jpg');
//        const status = await client.wait(jobId);

const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
const fs = require('fs');
const FormData = require('form-data');

class VFRClient {
  constructor(baseUrl) { this.baseUrl = baseUrl.replace(/\/$/, ''); }

  async tryOn(userPath, garmentPath, garmentSidePath) {
    const form = new FormData();
    form.append('user_image', fs.createReadStream(userPath));
    form.append('garment_front', fs.createReadStream(garmentPath));
    if (garmentSidePath) form.append('garment_side', fs.createReadStream(garmentSidePath));
    const r = await fetch(this.baseUrl + '/v1/jobs/tryon', { method: 'POST', body: form });
    if (!r.ok) throw new Error(await r.text());
    const js = await r.json();
    return js.job_id;
  }

  async status(jobId) {
    const r = await fetch(this.baseUrl + '/v1/jobs/' + jobId);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async wait(jobId, timeoutMs = 120000) {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const s = await this.status(jobId);
      if (s.status === 'completed' || s.status === 'failed') return s;
      await new Promise(res => setTimeout(res, 1000));
    }
    return { job_id: jobId, status: 'timeout' };
  }

  async download(jobId, outPath) {
    const r = await fetch(this.baseUrl + '/v1/jobs/' + jobId + '/result');
    if (!r.ok) throw new Error(await r.text());
    const buf = await r.buffer();
    fs.writeFileSync(outPath, buf);
    return outPath;
  }
}

module.exports = { VFRClient };

