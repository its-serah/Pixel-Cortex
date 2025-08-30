import React, { useState } from 'react';
import { useMutation } from 'react-query';
import api from '../utils/api';

function Section({ title, children }) {
  return (
    <div className="bg-white shadow rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3">{title}</h2>
      {children}
    </div>
  );
}

export default function AIWorkbench() {
  const [indexing, setIndexing] = useState(false);
  const [indexRes, setIndexRes] = useState(null);

  const [ragQuery, setRagQuery] = useState('VPN connection issue');
  const [ragRes, setRagRes] = useState(null);

  const [kgConcepts, setKgConcepts] = useState('VPN');
  const [kgRes, setKgRes] = useState(null);

  const [chatMsg, setChatMsg] = useState('How do I fix VPN timeouts?');
  const [chatRes, setChatRes] = useState(null);
  const [creating, setCreating] = useState(false);
  const [createdTicket, setCreatedTicket] = useState(null);
  const [updatingStatus, setUpdatingStatus] = useState(false);
  const [keepContext, setKeepContext] = useState(true);
  const [history, setHistory] = useState([]); // [{role:'user'|'assistant', content:string}]

  const indexPolicies = async () => {
    setIndexing(true);
    setIndexRes(null);
    try {
      const res = await api.post('/api/rag/index', {});
      setIndexRes(res.data);
    } catch (e) {
      setIndexRes({ error: e.response?.data?.detail || e.message });
    } finally {
      setIndexing(false);
    }
  };

  const createTicketFromChat = async () => {
    if (!chatMsg) return;
    setCreating(true);
    setCreatedTicket(null);
    try {
      // 1) Triage
      const triageBody = {
        title: chatMsg.slice(0, 120),
        description: chatRes?.response || chatMsg,
        user_role: 'user'
      };
      const triage = await api.post('/api/triage/analyze', triageBody).then(r => r.data);
      const category = triage?.triage_result?.category || 'other';
      const priority = triage?.triage_result?.priority || 'medium';

      // 2) Build ticket description with structured decision (if any)
      let descParts = [];
      descParts.push(`User Issue: ${chatMsg}`);
      if (chatRes?.structured) {
        const s = chatRes.structured;
        descParts.push('\nAI Decision:');
        descParts.push(`- Outcome: ${s.decision}`);
        if (s.decision_reason) descParts.push(`- Reason: ${s.decision_reason}`);
        if (Array.isArray(s.checklist) && s.checklist.length > 0) {
          descParts.push('\nChecklist:');
          s.checklist.forEach((step, i) => descParts.push(`  ${i + 1}. ${step}`));
        }
        const cites = s.citations_resolved?.length ? s.citations_resolved : s.policy_citations;
        if (Array.isArray(cites) && cites.length > 0) {
          descParts.push('\nCitations:');
          cites.forEach(c => descParts.push(`  ${c.reference || ''} ${c.title}`));
        }
        if (s.notes) {
          descParts.push(`\nNotes: ${s.notes}`);
        }
      } else if (chatRes?.response) {
        descParts.push(`\nAI Guidance:\n${chatRes.response}`);
      }
      const description = descParts.join('\n');

      // 3) Create ticket
      const createBody = {
        title: chatMsg.slice(0, 120),
        description,
        category,
        priority
      };
      const created = await api.post('/api/tickets', createBody).then(r => r.data);
      setCreatedTicket(created.ticket);
    } catch (e) {
      alert(`Ticket creation failed: ${e.response?.data?.detail || e.message}`);
    } finally {
      setCreating(false);
    }
  };

  const updateTicketStatus = async (status) => {
    if (!createdTicket?.id) return;
    setUpdatingStatus(true);
    try {
      // build policy_refs from structured citations if available
      let refs = [];
      const s = chatRes?.structured;
      if (s) {
        const cites = s.citations_resolved?.length ? s.citations_resolved : s.policy_citations;
        if (Array.isArray(cites)) {
          refs = cites.map(c => c.reference).filter(Boolean);
        }
      }
      const body = {
        status,
        resolution_code: status === 'resolved' || status === 'closed' ? 'AI-CHECKLIST-OK' : undefined,
        policy_refs: refs
      };
      const updated = await api.patch(`/api/tickets/${createdTicket.id}/status`, body).then(r => r.data);
      setCreatedTicket(updated.ticket);
    } catch (e) {
      alert(`Status update failed: ${e.response?.data?.detail || e.message}`);
    } finally {
      setUpdatingStatus(false);
    }
  };

  const doRagSearch = async () => {
    setRagRes(null);
    try {
      const res = await api.post('/api/rag/search', { query: ragQuery, k: 5 });
      setRagRes(res.data);
    } catch (e) {
      setRagRes({ error: e.response?.data?.detail || e.message });
    }
  };

  const doKgQuery = async () => {
    setKgRes(null);
    try {
      const cs = kgConcepts.split(',').map(s => s.trim()).filter(Boolean);
      const res = await api.post('/api/kg/query', { concepts: cs, max_hops: 2 });
      setKgRes(res.data);
    } catch (e) {
      setKgRes({ error: e.response?.data?.detail || e.message });
    }
  };

  const doChat = async () => {
    setChatRes(null);
    try {
      const body = { message: chatMsg, augment: true, k: 3 };
      if (keepContext && history.length > 0) {
        body.conversation_history = history.slice(-10);
      }
      const res = await api.post('/api/llm/chat', body);
      setChatRes(res.data);
      if (keepContext && res.data?.response) {
        setHistory(prev => [...prev, { role: 'user', content: chatMsg }, { role: 'assistant', content: res.data.response }]);
      }
    } catch (e) {
      setChatRes({ error: e.response?.data?.detail || e.message });
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">AI Workbench</h1>
        <button
          onClick={indexPolicies}
          disabled={indexing}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50"
        >
          {indexing ? 'Indexing...' : 'Index Policies'}
        </button>
      </div>

      {indexRes && (
        <Section title="Index Result">
          <pre className="text-xs bg-gray-50 p-3 rounded overflow-auto">{JSON.stringify(indexRes, null, 2)}</pre>
        </Section>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Section title="RAG Search">
          <div className="flex space-x-2 mb-3">
            <input
              className="flex-1 border rounded px-3 py-2"
              placeholder="Search query"
              value={ragQuery}
              onChange={(e) => setRagQuery(e.target.value)}
            />
            <button onClick={doRagSearch} className="px-4 py-2 rounded bg-gray-900 text-white">Search</button>
          </div>
          {ragRes && (
            <div className="space-y-3">
              <div className="text-sm text-gray-700 whitespace-pre-wrap">{ragRes.summary}</div>
              <ul className="space-y-2">
                {ragRes.results?.map((r, i) => (
                  <li key={r.chunk_id} className="p-3 border rounded">
                    <div className="text-xs text-gray-500">[{i+1}] {r.document_title} (score: {r.score?.toFixed(3)})</div>
                    <div className="text-sm mt-1">{r.content.slice(0, 300)}...</div>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Section>

        <Section title="KG Query">
          <div className="flex space-x-2 mb-3">
            <input
              className="flex-1 border rounded px-3 py-2"
              placeholder="Concepts (comma-separated)"
              value={kgConcepts}
              onChange={(e) => setKgConcepts(e.target.value)}
            />
            <button onClick={doKgQuery} className="px-4 py-2 rounded bg-gray-900 text-white">Query</button>
          </div>
          {kgRes && (
            <div className="space-y-3">
              {kgRes.related && (
                <div>
                  <div className="text-sm font-medium mb-1">Related Concepts:</div>
                  <div className="text-sm text-gray-700">{kgRes.related.join(', ') || 'None'}</div>
                </div>
              )}
              {kgRes.hops && (
                <div>
                  <div className="text-sm font-medium mb-1">Traversal Hops:</div>
                  <ul className="text-xs text-gray-700 list-disc pl-4 space-y-1">
                    {kgRes.hops.map((h, idx) => (
                      <li key={idx}>{h.from} → {h.to} (depth {h.depth}, rel {h.relationship})</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </Section>
      </div>

      <Section title="LLM Chat (RAG-augmented)">
        <div className="flex items-center space-x-2 mb-3">
          <input
            className="flex-1 border rounded px-3 py-2"
            placeholder="Type a question..."
            value={chatMsg}
            onChange={(e) => setChatMsg(e.target.value)}
          />
          <label className="flex items-center space-x-2 text-xs text-gray-700">
            <input type="checkbox" checked={keepContext} onChange={(e)=>setKeepContext(e.target.checked)} />
            <span>Keep context</span>
          </label>
          <button onClick={doChat} className="px-4 py-2 rounded bg-primary-600 text-white">Ask</button>
          <button onClick={createTicketFromChat} disabled={!chatMsg || creating} className="px-4 py-2 rounded bg-green-600 text-white disabled:opacity-50">{creating ? 'Creating…' : 'Create Ticket'}</button>
        </div>
        {keepContext && history.length > 0 && (
          <div className="mb-4 bg-white border rounded p-3 max-h-64 overflow-auto">
            <div className="text-sm font-semibold mb-2">Conversation</div>
            <ul className="space-y-2">
              {history.map((m, idx) => (
                <li key={idx} className="text-sm">
                  <span className={`font-medium ${m.role==='user'?'text-blue-700':'text-green-700'}`}>{m.role==='user'?'User':'Assistant'}:</span>
                  <span className="ml-2 whitespace-pre-wrap">{m.content}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {chatRes && (
          <>
            <div className="text-sm whitespace-pre-wrap bg-gray-50 p-3 rounded mb-4">
              {chatRes.response || chatRes.error}
            </div>

            {chatRes.structured && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white border rounded p-3">
                  <div className="text-sm font-semibold mb-2">Decision</div>
                  <div className="text-sm"><span className="font-medium">Outcome:</span> {chatRes.structured.decision || 'n/a'}</div>
                  {chatRes.structured.decision_reason && (
                    <div className="text-sm mt-1"><span className="font-medium">Reason:</span> {chatRes.structured.decision_reason}</div>
                  )}
                </div>

                <div className="bg-white border rounded p-3">
                  <div className="text-sm font-semibold mb-2">Checklist</div>
                  {Array.isArray(chatRes.structured.checklist) && chatRes.structured.checklist.length > 0 ? (
                    <ol className="list-decimal pl-5 text-sm space-y-1">
                      {chatRes.structured.checklist.map((s, idx) => (
                        <li key={idx}>{s}</li>
                      ))}
                    </ol>
                  ) : (
                    <div className="text-sm text-gray-500">No steps provided.</div>
                  )}
                </div>

                <div className="bg-white border rounded p-3">
                  <div className="text-sm font-semibold mb-2">Citations</div>
                  {Array.isArray(chatRes.structured.citations_resolved) && chatRes.structured.citations_resolved.length > 0 ? (
                    <ul className="text-sm space-y-1">
                      {chatRes.structured.citations_resolved.map((c, idx) => (
                        <li key={idx}>
                          <span className="font-mono mr-1">{c.reference}</span>
                          {c.title}
                          {c.chunk_id ? <span className="text-gray-500"> (chunk {c.chunk_id.slice(0,6)}…)</span> : null}
                        </li>
                      ))}
                    </ul>
                  ) : Array.isArray(chatRes.structured.policy_citations) && chatRes.structured.policy_citations.length > 0 ? (
                    <ul className="text-sm space-y-1">
                      {chatRes.structured.policy_citations.map((c, idx) => (
                        <li key={idx}>
                          <span className="font-mono mr-1">{c.reference}</span>
                          {c.title}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="text-sm text-gray-500">No citations.</div>
                  )}
                </div>

                <div className="bg-white border rounded p-3">
                  <div className="text-sm font-semibold mb-2">Notes</div>
                  <div className="text-sm whitespace-pre-wrap">{chatRes.structured.notes || '—'}</div>
                </div>
              </div>
            )}
          </>

            {createdTicket && (
              <div className="mt-4 bg-white border rounded p-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-sm font-semibold">Created Ticket</div>
                    <div className="text-sm text-gray-700">#{createdTicket.id} — {createdTicket.title}</div>
                    <div className="text-xs text-gray-500 mt-1 capitalize">Status: {createdTicket.status}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button disabled={updatingStatus || createdTicket.status === 'in_progress'} onClick={() => updateTicketStatus('in_progress')} className="px-3 py-1 text-xs rounded border bg-white hover:bg-gray-50 disabled:opacity-50">In Progress</button>
                    <button disabled={updatingStatus || createdTicket.status === 'resolved' || createdTicket.status === 'closed'} onClick={() => updateTicketStatus('resolved')} className="px-3 py-1 text-xs rounded border bg-white hover:bg-gray-50 disabled:opacity-50">Resolve</button>
                    <button disabled={updatingStatus || createdTicket.status === 'closed'} onClick={() => updateTicketStatus('closed')} className="px-3 py-1 text-xs rounded border bg-white hover:bg-gray-50 disabled:opacity-50">Close</button>
                  </div>
                </div>
              </div>
            )}
        )}
      </Section>
    </div>
  );
}

