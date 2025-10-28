import React, { useState } from 'react'

const App: React.FC = () => {
  const [user, setUser] = useState<File | null>(null)
  const [garment, setGarment] = useState<File | null>(null)
  const [garmentSide, setGarmentSide] = useState<File | null>(null)
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:8000')
  const [log, setLog] = useState('')
  const [resultUrl, setResultUrl] = useState<string | null>(null)

  const append = (m: string) => setLog(s => s + m + '\n')

  const submit = async () => {
    if (!user || !garment) {
      append('Select user and garment images.')
      return
    }
    const form = new FormData()
    form.append('user_image', user)
    form.append('garment_front', garment)
    if (garmentSide) form.append('garment_side', garmentSide)
    append('Submitting job...')
    const r = await fetch(baseUrl.replace(/\/$/, '') + '/v1/jobs/tryon', { method: 'POST', body: form })
    if (!r.ok) {
      append('Submit failed: ' + (await r.text()))
      return
    }
    const { job_id } = await r.json()
    append('Job ID: ' + job_id)
    let status = 'queued'
    while (status !== 'completed' && status !== 'failed') {
      await new Promise(res => setTimeout(res, 1000))
      const s = await fetch(baseUrl.replace(/\/$/, '') + '/v1/jobs/' + job_id)
      const js = await s.json()
      status = js.status
      append('Status: ' + status)
      if (status === 'completed') {
        const url = baseUrl.replace(/\/$/, '') + '/v1/jobs/' + job_id + '/result'
        setResultUrl(url)
      }
    }
  }

  return (
    <div style={{ padding: 16, fontFamily: 'system-ui' }}>
      <h1>Neorose VFR (React)</h1>
      <div style={{ marginBottom: 8 }}>
        <label>User Image</label>
        <input type="file" accept="image/*" onChange={e => setUser(e.target.files?.[0] || null)} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <label>Garment Front</label>
        <input type="file" accept="image/*" onChange={e => setGarment(e.target.files?.[0] || null)} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <label>Garment Side (optional)</label>
        <input type="file" accept="image/*" onChange={e => setGarmentSide(e.target.files?.[0] || null)} />
      </div>
      <div style={{ marginBottom: 8 }}>
        <input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} style={{ width: 320 }} />
      </div>
      <button onClick={submit}>Submit</button>
      <pre style={{ whiteSpace: 'pre-wrap', background: '#f7f7f7', padding: 8, marginTop: 12 }}>{log}</pre>
      {resultUrl && (
        <div>
          <h3>Result</h3>
          <img src={resultUrl} style={{ maxWidth: '100%', border: '1px solid #ddd', borderRadius: 6 }} />
        </div>
      )}
    </div>
  )
}

export default App

