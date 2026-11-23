import { useEffect, useRef, useState } from 'react'
import {
  Activity, ArrowUpRight, BarChart3, Bell, CheckCircle2,
  ChevronDown, CircleHelp, Clock3, Cpu, Database, Download,
  Eraser, FileImage, FlaskConical, Gauge, LayoutDashboard,
  Menu, Moon, MoreHorizontal, Pencil, Settings, ShieldCheck,
  Sparkles, Trash2, Undo2, X, Zap
} from 'lucide-react'
import { recognizeSequence, type SequencePrediction } from './services/recognition'
import './App.css'

type Page = 'Overview' | 'Handwriting Studio' | 'Dataset' | 'Evaluations' | 'Models' | 'Settings'

const navigation: { name: Page; icon: typeof LayoutDashboard; group: string }[] = [
  { name: 'Overview', icon: LayoutDashboard, group: 'WORKSPACE' },
  { name: 'Handwriting Studio', icon: Pencil, group: 'WORKSPACE' },
  { name: 'Dataset', icon: Database, group: 'WORKSPACE' },
  { name: 'Evaluations', icon: BarChart3, group: 'INTELLIGENCE' },
  { name: 'Models', icon: Cpu, group: 'INTELLIGENCE' },
  { name: 'Settings', icon: Settings, group: 'SYSTEM' },
]

function App() {
  const [page, setPage] = useState<Page>('Overview')
  const [mobileMenu, setMobileMenu] = useState(false)
  const [dark, setDark] = useState(false)
  const [brush, setBrush] = useState(8)
  const [erasing, setErasing] = useState(false)
  const [groundTruth, setGroundTruth] = useState('')
  const [writer, setWriter] = useState('W01')
  const [notice, setNotice] = useState('')
  const [autoDetect, setAutoDetect] = useState(true)
  const [recognition, setRecognition] = useState<SequencePrediction | null>(null)
  const [recognitionStatus, setRecognitionStatus] = useState('idle')
  const [recognitionError, setRecognitionError] = useState('')
  const [recognitionLatency, setRecognitionLatency] = useState<number | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const drawingRef = useRef(false)
  const undoRef = useRef<ImageData[]>([])
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const requestIdRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
  }, [page])

  async function recognizeCanvas() {
    const canvas = canvasRef.current
    if (!canvas) return

    if (drawingRef.current) {
      setRecognitionError('Finish writing before recognition.')
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data
    let hasInk = false

    for (let i = 0; i < pixels.length; i += 4) {
      if (pixels[i] < 245 || pixels[i + 1] < 245 || pixels[i + 2] < 245) {
        hasInk = true
        break
      }
    }

    if (!hasInk) {
      setRecognition(null)
      setRecognitionStatus('idle')
      setRecognitionError('Write some digits on the canvas first.')
      return
    }

    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
      debounceRef.current = null
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    const requestId = ++requestIdRef.current
    const startedAt = performance.now()

    setRecognitionStatus('processing')
    setRecognitionError('')

    try {
      const blob = await new Promise<Blob | null>(resolve => {
        canvas.toBlob(resolve, 'image/png')
      })

      if (!blob) throw new Error('Could not capture the canvas image.')
      if (requestId !== requestIdRef.current) return

      const result = await recognizeSequence(blob, controller.signal)

      if (requestId !== requestIdRef.current) return

      setRecognition(result)
      setRecognitionLatency(performance.now() - startedAt)
      setRecognitionStatus(result.status)
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') return
      if (requestId !== requestIdRef.current) return

      setRecognitionStatus('error')
      setRecognitionError(
        error instanceof Error ? error.message : 'Recognition failed.',
      )
    }
  }

  function scheduleAutomaticRecognition() {
    if (!autoDetect) return

    if (debounceRef.current) clearTimeout(debounceRef.current)

    abortRef.current?.abort()
    requestIdRef.current += 1

    setRecognitionStatus('waiting')
    setRecognitionError('')

    debounceRef.current = setTimeout(() => {
      debounceRef.current = null
      void recognizeCanvas()
    }, 800)
  }

  function cancelRecognition() {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
      debounceRef.current = null
    }

    abortRef.current?.abort()
    requestIdRef.current += 1
    setRecognitionStatus('idle')
    setRecognition(null)
    setRecognitionError('')
    setRecognitionLatency(null)
  }

  function canvasPosition(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current!
    const rect = canvas.getBoundingClientRect()
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height,
    }
  }

  function startDrawing(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    event.preventDefault()
    undoRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height))
    if (undoRef.current.length > 25) undoRef.current.shift()
    drawingRef.current = true
    canvas.setPointerCapture(event.pointerId)
    const p = canvasPosition(event)
    ctx.beginPath()
    ctx.moveTo(p.x, p.y)
    ctx.lineTo(p.x + 0.1, p.y + 0.1)
    ctx.strokeStyle = erasing ? '#ffffff' : '#111827'
    ctx.lineWidth = erasing ? brush * 3 : brush
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.stroke()
  }

  function draw(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!drawingRef.current) return
    event.preventDefault()
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!ctx) return
    const p = canvasPosition(event)
    ctx.lineTo(p.x, p.y)
    ctx.stroke()
  }

  function stopDrawing() {
    if (!drawingRef.current) return

    drawingRef.current = false
    const ctx = canvasRef.current?.getContext('2d')
    ctx?.beginPath()

    scheduleAutomaticRecognition()
  }

  function clearCanvas() {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    cancelRecognition()
    setRecognitionError('')
    undoRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height))
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.beginPath()
  }

  function undoCanvas() {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    const previous = undoRef.current.pop()
    if (canvas && ctx && previous) ctx.putImageData(previous, 0, 0)
  }

  function exportCanvas() {
    const canvas = canvasRef.current
    if (!canvas) return
    if (!groundTruth.match(/^\d{2,6}$/)) {
      setNotice('Enter the correct sequence (2–6 digits) before exporting.')
      return
    }
    const link = document.createElement('a')
    link.download = `numeriq-${writer}-${groundTruth}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
    setNotice('PNG exported. Dataset upload integration is the next backend milestone.')
  }

  function navigate(next: Page) {
    setPage(next)
    setMobileMenu(false)
    setNotice('')
  }

  const groups = ['WORKSPACE', 'INTELLIGENCE', 'SYSTEM']

  return (
    <div className={`app-shell ${dark ? 'dark-theme' : ''}`}>
      {mobileMenu && <button className="mobile-overlay" onClick={() => setMobileMenu(false)} aria-label="Close navigation" />}
      <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`}>
        <div className="brand">
          <div className="brand-mark"><Sparkles size={21} /></div>
          <div className="brand-copy">
            <strong>numeriq<span>.</span></strong>
            <small>INTELLIGENCE PLATFORM</small>
          </div>
          <button className="icon-button sidebar-close" onClick={() => setMobileMenu(false)} aria-label="Close menu"><X size={18}/></button>
        </div>

        <div className="workspace-select">
          <div className="workspace-avatar">N</div>
          <div className="workspace-copy"><strong>NumerIQ Workspace</strong><small>Personal workspace</small></div>
          <ChevronDown size={15} />
        </div>

        <nav className="navigation">
          {groups.map(group => (
            <div className="nav-group" key={group}>
              <div className="nav-label">{group}</div>
              {navigation.filter(item => item.group === group).map(item => {
                const Icon = item.icon
                return (
                  <button key={item.name} onClick={() => navigate(item.name)}
                    className={`nav-item ${page === item.name ? 'active' : ''}`}>
                    <Icon size={18} strokeWidth={1.8}/>
                    <span>{item.name}</span>
                    {item.name === 'Handwriting Studio' && <span className="nav-new">NEW</span>}
                  </button>
                )
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="plan-card">
            <div className="plan-icon"><Zap size={17}/></div>
            <strong>AI workspace</strong>
            <p>Your local intelligence environment is ready for development.</p>
            <div className="plan-status"><span/> Local development</div>
          </div>
          <div className="profile">
            <div className="profile-avatar">N</div>
            <div className="profile-copy"><strong>NumerIQ Admin</strong><small>Workspace owner</small></div>
            <MoreHorizontal size={19}/>
          </div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu-button" onClick={() => setMobileMenu(true)} aria-label="Open navigation"><Menu size={20}/></button>
            <div className="breadcrumbs"><span>Workspace</span><span className="crumb-separator">/</span><strong>{page}</strong></div>
          </div>
          <div className="topbar-actions">
            <div className="environment"><span/> LOCAL ENVIRONMENT</div>
            <button className="icon-button" onClick={() => setDark(!dark)} aria-label="Toggle theme"><Moon size={18}/></button>
            <button className="icon-button notification-button" aria-label="Notifications"><Bell size={18}/><i/></button>
            <div className="top-avatar">N</div>
          </div>
        </header>

        <div className="page-content">
          {page === 'Overview' && <Overview onNavigate={navigate}/>}
          {page === 'Handwriting Studio' && (
            <section>
              <PageHeading eyebrow="DATA COLLECTION" title="Handwriting Studio" subtitle="Capture handwritten sequences and prepare samples for independent model evaluation."/>
              <div className="studio-layout">
                <div className="panel drawing-panel">
                  <div className="panel-heading">
                    <div><h3>Drawing workspace</h3><p>Write naturally from left to right.</p></div>
                    <span className="live-indicator"><i/> CANVAS READY</span>
                  </div>
                  <div className="drawing-paper">
                    <div className="paper-guides"><span/><span/><span/></div>
                    <canvas ref={canvasRef} width={1200} height={480}
                      onPointerDown={startDrawing} onPointerMove={draw}
                      onPointerUp={stopDrawing} onPointerCancel={stopDrawing}
                      onLostPointerCapture={stopDrawing}/>
                  </div>
                  <div className="canvas-toolbar">
                    <div className="tool-group">
                      <button className={`tool-button ${!erasing ? 'selected' : ''}`} onClick={() => setErasing(false)}><Pencil size={16}/> Pen</button>
                      <button className={`tool-button ${erasing ? 'selected' : ''}`} onClick={() => setErasing(true)}><Eraser size={16}/> Eraser</button>
                    </div>
                    <div className="tool-group">
                      <label className="brush-label">SIZE <input type="range" min="3" max="24" value={brush} onChange={e => setBrush(Number(e.target.value))}/></label>
                      <button className="tool-button" onClick={undoCanvas}><Undo2 size={16}/> Undo</button>
                      <button className="tool-button danger-tool" onClick={clearCanvas}><Trash2 size={16}/> Clear</button>
                    </div>
                  </div>
                  <div className="canvas-tip"><CircleHelp size={15}/> Use a mouse, touchscreen or stylus. Draw with comfortable spacing between digits.</div>
                </div>
                <div className="panel sample-panel">
                  <div className="panel-heading"><div><h3>Sample details</h3><p>Label your handwriting accurately.</p></div><FileImage size={20} className="muted-icon"/></div>
                  <div className="recognition-controls">
                    <div className="recognition-mode">
                      <div>
                        <strong>Automatic detection</strong>
                        <small>Recognize after you pause writing</small>
                      </div>
                      <label className="switch-control">
                        <input
                          type="checkbox"
                          checked={autoDetect}
                          onChange={event => {
                            const enabled = event.target.checked
                            setAutoDetect(enabled)
                            if (!enabled && debounceRef.current) {
                              clearTimeout(debounceRef.current)
                              debounceRef.current = null
                              if (recognitionStatus === 'waiting') {
                                setRecognitionStatus('idle')
                              }
                            }
                          }}
                          aria-label="Enable automatic detection"
                        />
                        <span className="switch-slider"/>
                      </label>
                    </div>

                    <button
                      className="button-primary full-button"
                      onClick={() => void recognizeCanvas()}
                      disabled={recognitionStatus === 'processing'}
                    >
                      <Zap size={16}/>
                      {recognitionStatus === 'processing' ? 'Recognizing...' : 'Recognize now'}
                    </button>

                    <div className="recognition-result" aria-live="polite">
                      <div className="result-heading">
                        <strong>Recognition result</strong>
                        {recognitionLatency !== null && (
                          <span>{recognitionLatency.toFixed(1)} ms</span>
                        )}
                      </div>

                      {recognitionStatus === 'waiting' && (
                        <p className="result-message">Writing detected. Waiting for a pause...</p>
                      )}
                      {recognitionStatus === 'processing' && (
                        <p className="result-message">Analyzing your handwriting with the V2 engine...</p>
                      )}
                      {recognitionError && (
                        <p className="result-error">{recognitionError}</p>
                      )}
                      {recognition && (
                        <>
                          <div className="recognized-sequence">
                            {recognition.sequence || 'No digits detected'}
                          </div>
                          <div className="result-meta">
                            {recognition.digit_count} digit(s) detected
                            <span>{recognition.status.replaceAll('_', ' ')}</span>
                          </div>
                          {recognition.digits.length > 0 && (
                            <div className="digit-results">
                              {recognition.digits.map(item => (
                                <div className="digit-result" key={item.position}>
                                  <strong>{item.digit}</strong>
                                  <span>{(item.confidence * 100).toFixed(1)}%</span>
                                  {item.is_uncertain && <small>Uncertain</small>}
                                </div>
                              ))}
                            </div>
                          )}
                          {recognition.warnings.map((warning, index) => (
                            <p className="result-warning" key={index}>{warning}</p>
                          ))}
                        </>
                      )}
                      {!recognition && !recognitionError &&
                        recognitionStatus === 'idle' && (
                          <p className="result-message">Your prediction will appear here.</p>
                        )}
                    </div>
                  </div>
                  <div className="form-field"><label>GROUND TRUTH <span className="required">*</span></label><input value={groundTruth} onChange={e => setGroundTruth(e.target.value.replace(/\D/g, '').slice(0, 6))} placeholder="e.g. 48291" inputMode="numeric"/><small>Enter the exact digits you wrote (2–6 digits).</small></div>
                  <div className="form-field"><label>WRITER ID <span className="required">*</span></label><input value={writer} onChange={e => setWriter(e.target.value)} placeholder="e.g. W01"/><small>Use a consistent anonymous ID for each writer.</small></div>
                  <div className="form-field"><label>CAPTURE METHOD</label><div className="readonly-field">Digital canvas <span className="tag">DIGITAL</span></div></div>
                  <div className="form-field"><label>WRITING INSTRUMENT</label><div className="readonly-field">Mouse / Touch / Stylus</div></div>
                  <div className="sample-note"><ShieldCheck size={17}/><span><strong>Evaluation-only collection</strong><br/>These samples are not automatically used for model training.</span></div>
                  <button className="button-primary full-button" onClick={exportCanvas}><Download size={17}/> Export PNG sample</button>
                  {notice && <div className="notice">{notice}</div>}
                  <p className="export-note">PNG export is local. Server-side dataset saving will be enabled after backend integration.</p>
                </div>
              </div>
            </section>
          )}
          {page === 'Dataset' && <PlaceholderPage icon={Database} eyebrow="DATA MANAGEMENT" title="Dataset" description="Manage handwriting samples, writer metadata, dataset quality and evaluation partitions." items={['Sample gallery and image previews','Writer-level dataset statistics','Manifest validation and duplicate detection','Evaluation-only dataset controls']}/>}
          {page === 'Evaluations' && <PlaceholderPage icon={FlaskConical} eyebrow="MODEL VALIDATION" title="Evaluations" description="Measure recognition quality using independent labeled handwriting samples." items={['Exact sequence accuracy and CER','Segmentation mismatch and no-detection rates','P50 and P95 inference latency','Per-writer performance reports']}/>}
          {page === 'Models' && <PlaceholderPage icon={Cpu} eyebrow="MODEL REGISTRY" title="Models" description="Inspect available recognition engines and their verified checkpoints." items={['Legacy MNIST CNN — epoch 6','CNN robustness V2 — epoch 9','Model metadata and validation history','Checkpoint integrity and version tracking']}/>}
          {page === 'Settings' && <PlaceholderPage icon={Settings} eyebrow="PREFERENCES" title="Settings" description="Configure your local NumerIQ development workspace." items={['Application preferences','Inference engine configuration','Dataset storage settings','API connection and diagnostics']}/>}
        </div>
        <footer className="footer"><span>© 2026 NumerIQ Intelligence</span><span><span className="footer-dot"/> Local development build <span className="footer-divider">·</span> v0.1.0</span></footer>
      </main>
    </div>
  )
}

function PageHeading({eyebrow, title, subtitle}: {eyebrow:string; title:string; subtitle:string}) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div></div>
}

function StatCard({label, value, detail, icon: Icon, tone='blue'}: {label:string; value:string; detail:string; icon:typeof Activity; tone?:string}) {
  return <div className="stat-card">
    <div className="stat-top"><span>{label}</span><div className={`stat-icon ${tone}`}><Icon size={18}/></div></div>
    <div className="stat-value">{value}</div><div className="stat-detail">{detail}</div>
  </div>
}

function Overview({onNavigate}: {onNavigate:(page:Page)=>void}) {
  return <>
    <PageHeading eyebrow="OVERVIEW" title="Good to see you." subtitle="Your handwriting intelligence workspace at a glance."/>
    <div className="welcome-banner">
      <div className="welcome-content"><div className="welcome-kicker"><Sparkles size={14}/> NUMERIQ INTELLIGENCE</div><h2>Turn handwriting into<br/><span>intelligent data.</span></h2><p>Explore recognition, collect handwriting samples and evaluate model performance in one workspace.</p><button onClick={() => onNavigate('Handwriting Studio')} className="welcome-button">Open handwriting studio <ArrowUpRight size={16}/></button></div>
      <div className="welcome-art"><div className="art-orbit orbit-one"/><div className="art-orbit orbit-two"/><div className="art-core"><Sparkles size={38}/></div><div className="art-chip chip-one"><Activity size={15}/> AI recognition</div><div className="art-chip chip-two"><CheckCircle2 size={15}/> V2 engine</div><div className="art-chip chip-three">01 — 09</div></div>
    </div>

    <div className="section-heading"><div><h2>Performance snapshot</h2><p>Previously measured synthetic benchmark results</p></div><span className="benchmark-tag"><span/> SYNTHETIC BENCHMARK</span></div>
    <div className="stats-grid">
      <StatCard label="Exact sequence accuracy" value="89.70%" detail="1,000 synthetic sequences" icon={Gauge} tone="violet"/>
      <StatCard label="Character error rate" value="2.675%" detail="Lower is better" icon={Activity} tone="green"/>
      <StatCard label="Mean latency" value="4.883 ms" detail="End-to-end benchmark" icon={Zap} tone="amber"/>
      <StatCard label="P95 latency" value="6.802 ms" detail="Measured on CPU" icon={Clock3} tone="blue"/>
    </div>

    <div className="content-grid">
      <div className="panel benchmark-panel">
        <div className="panel-heading"><div><h3>Recognition benchmark</h3><p>Current verified synthetic sequence results</p></div><button className="subtle-button" onClick={() => onNavigate('Evaluations')}>View evaluations <ArrowUpRight size={14}/></button></div>
        <div className="benchmark-main"><div><div className="big-metric">89.70<span>%</span></div><div className="metric-caption">Exact sequence accuracy</div></div><div className="benchmark-status"><CheckCircle2 size={16}/> Benchmark completed</div></div>
        <div className="progress-track"><div className="progress-fill"/></div>
        <div className="benchmark-foot"><span>897 exact matches</span><span>1,000 samples</span></div>
        <div className="divider"/>
        <div className="mini-metrics"><div><span>Character error rate</span><strong>2.675%</strong></div><div><span>Count mismatch</span><strong>1.00%</strong></div><div><span>Processing failures</span><strong>Not reported here</strong></div></div>
        <div className="disclaimer"><ShieldCheck size={15}/> Synthetic MNIST-composed data. Natural handwriting accuracy has not yet been established.</div>
      </div>
      <div className="panel engine-panel">
        <div className="panel-heading"><div><h3>Recognition engine</h3><p>Local model configuration</p></div><Cpu size={20} className="muted-icon"/></div>
        <div className="engine-visual"><div className="engine-ring"><div><Cpu size={28}/><strong>V2</strong></div></div><div className="engine-name">CNN Robustness V2</div><div className="engine-subtitle">DigitCNN · PyTorch</div></div>
        <div className="engine-details"><div><span>Best validation accuracy</span><strong>96.14%</strong></div><div><span>Best epoch</span><strong>9</strong></div><div><span>Inference device</span><strong><span className="device-dot"/> CPU</strong></div><div><span>Model status</span><strong className="status-ready">Checkpoint verified</strong></div></div>
        <button className="outline-button full-button" onClick={() => onNavigate('Models')}>Explore model details <ArrowUpRight size={15}/></button>
      </div>
    </div>

    <div className="section-heading lower-heading"><div><h2>Quick actions</h2><p>Continue building your handwriting intelligence workflow.</p></div></div>
    <div className="quick-grid">
      <button className="quick-card" onClick={() => onNavigate('Handwriting Studio')}><div className="quick-icon purple"><Pencil size={19}/></div><div><strong>Collect handwriting</strong><span>Open the digital writing canvas</span></div><ArrowUpRight size={17}/></button>
      <button className="quick-card" onClick={() => onNavigate('Dataset')}><div className="quick-icon teal"><Database size={19}/></div><div><strong>Manage dataset</strong><span>Review samples and metadata</span></div><ArrowUpRight size={17}/></button>
      <button className="quick-card" onClick={() => onNavigate('Evaluations')}><div className="quick-icon orange"><BarChart3 size={19}/></div><div><strong>Evaluate model</strong><span>Inspect recognition performance</span></div><ArrowUpRight size={17}/></button>
    </div>
  </>
}

function PlaceholderPage({icon:Icon, eyebrow, title, description, items}: {icon:typeof Activity; eyebrow:string; title:string; description:string; items:string[]}) {
  return <section><PageHeading eyebrow={eyebrow} title={title} subtitle={description}/><div className="panel placeholder-panel"><div className="placeholder-icon"><Icon size={27}/></div><h2>{title} workspace</h2><p>This module is part of the NumerIQ application roadmap. Its interface and backend workflow will be implemented in the next development milestones.</p><div className="placeholder-list">{items.map(item => <div key={item}><CheckCircle2 size={16}/>{item}</div>)}</div><div className="coming-soon"><span/> MODULE FOUNDATION PENDING</div></div></section>
}

export default App
