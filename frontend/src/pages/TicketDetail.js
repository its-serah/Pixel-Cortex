import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { useQuery } from 'react-query';
import { Clock, User, Tag, AlertCircle, Brain, ChevronDown, ChevronRight } from 'lucide-react';
import api from '../utils/api';

function ExplanationPanel({ explanation }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [activeSection, setActiveSection] = useState('reasoning');

  if (!explanation) return null;

  const sections = [
    { id: 'reasoning', label: 'Reasoning Trace', icon: Brain },
    { id: 'citations', label: 'Policy Citations', icon: Tag },
    { id: 'alternatives', label: 'Alternatives', icon: AlertCircle },
    { id: 'telemetry', label: 'Performance', icon: Clock },
  ];

  return (
    <div className="bg-white shadow rounded-lg">
      <div className="px-4 py-5 sm:px-6">
        <div className="flex items-center justify-between">
          <h3 className="text-lg leading-6 font-medium text-gray-900 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-primary-500" />
            AI Explanation
          </h3>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-gray-400 hover:text-gray-600"
          >
            {isExpanded ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
          </button>
        </div>
        
        {/* Summary */}
        <div className="mt-2">
          <p className="text-sm text-gray-600">{explanation.answer}</p>
          <div className="mt-2 flex items-center space-x-4">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Confidence: {(explanation.confidence * 100).toFixed(1)}%
            </span>
            <span className="text-xs text-gray-500">
              Processed in {explanation.telemetry?.latency_ms}ms
            </span>
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-gray-200">
          {/* Section tabs */}
          <div className="px-4 py-3 border-b border-gray-200">
            <nav className="flex space-x-8">
              {sections.map((section) => {
                const Icon = section.icon;
                return (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={`${
                      activeSection === section.id
                        ? 'text-primary-600 border-primary-600'
                        : 'text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300'
                    } flex items-center py-2 px-1 border-b-2 font-medium text-sm`}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {section.label}
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Section content */}
          <div className="px-4 py-5">
            {activeSection === 'reasoning' && (
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Decision Process</h4>
                {explanation.reasoning_trace?.map((step, index) => (
                  <div key={index} className="flex space-x-3">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                        <span className="text-sm font-medium text-primary-600">{step.step}</span>
                      </div>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900">{step.action}</p>
                      <p className="text-sm text-gray-600">{step.rationale}</p>
                      <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-700 mt-1">
                        Confidence: {(step.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeSection === 'citations' && (
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Policy References</h4>
                {explanation.policy_citations?.length > 0 ? (
                  explanation.policy_citations.map((citation, index) => (
                    <div key={index} className="border rounded-lg p-3 bg-gray-50">
                      <div className="flex justify-between items-start">
                        <h5 className="font-medium text-gray-900">{citation.document_title}</h5>
                        <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                          Score: {(citation.relevance_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mt-2">{citation.chunk_content}</p>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No policy citations available</p>
                )}
              </div>
            )}

            {activeSection === 'alternatives' && (
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Alternative Options</h4>
                {explanation.alternatives_considered?.length > 0 ? (
                  explanation.alternatives_considered.map((alt, index) => (
                    <div key={index} className="border rounded-lg p-3">
                      <h5 className="font-medium text-gray-900">{alt.option}</h5>
                      <div className="mt-2 grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs font-medium text-green-700">Pros:</p>
                          <ul className="text-xs text-gray-600 list-disc list-inside">
                            {alt.pros?.map((pro, i) => <li key={i}>{pro}</li>)}
                          </ul>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-red-700">Cons:</p>
                          <ul className="text-xs text-gray-600 list-disc list-inside">
                            {alt.cons?.map((con, i) => <li key={i}>{con}</li>)}
                          </ul>
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No alternatives considered</p>
                )}
              </div>
            )}

            {activeSection === 'telemetry' && (
              <div className="space-y-4">
                <h4 className="font-medium text-gray-900">Performance Metrics</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <p className="text-xs font-medium text-gray-700">Total Latency</p>
                    <p className="text-lg font-semibold text-gray-900">{explanation.telemetry?.latency_ms}ms</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <p className="text-xs font-medium text-gray-700">Policy Chunks</p>
                    <p className="text-lg font-semibold text-gray-900">{explanation.telemetry?.total_chunks_considered}</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <p className="text-xs font-medium text-gray-700">Triage Time</p>
                    <p className="text-lg font-semibold text-gray-900">{explanation.telemetry?.triage_time_ms}ms</p>
                  </div>
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <p className="text-xs font-medium text-gray-700">Planning Time</p>
                    <p className="text-lg font-semibold text-gray-900">{explanation.telemetry?.planning_time_ms}ms</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function TicketDetail() {
  const { id } = useParams();
  
  const { data: ticket, isLoading, error } = useQuery(
    ['ticket', id],
    () => api.get(`/api/tickets/${id}`).then(res => res.data)
  );
  
  const { data: explanation } = useQuery(
    ['ticket-explanation', id],
    () => api.get(`/api/tickets/${id}/explanation`).then(res => res.data),
    { enabled: !!ticket }
  );

  if (isLoading) return <div className="text-center py-8">Loading...</div>;
  if (error) return <div className="text-center py-8 text-red-600">Error loading ticket</div>;

  const priorityColors = {
    low: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800'
  };

  const statusColors = {
    open: 'bg-blue-100 text-blue-800',
    in_progress: 'bg-yellow-100 text-yellow-800',
    waiting_for_user: 'bg-purple-100 text-purple-800',
    resolved: 'bg-green-100 text-green-800',
    closed: 'bg-gray-100 text-gray-800'
  };

  return (
    <div className="space-y-6">
      {/* Ticket Header */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">#{ticket.id} {ticket.title}</h1>
              <div className="mt-2 flex items-center space-x-4">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColors[ticket.status]}`}>
                  {ticket.status.replace('_', ' ')}
                </span>
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${priorityColors[ticket.priority]}`}>
                  {ticket.priority}
                </span>
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                  {ticket.category}
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Created</p>
              <p className="text-sm font-medium">{new Date(ticket.created_at).toLocaleDateString()}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Ticket Details */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900">Description</h3>
              <div className="mt-4">
                <p className="text-gray-700 whitespace-pre-wrap">{ticket.description}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Explanation Panel - Hidden from users, only admins can access */}
        <div className="space-y-6">
          {/* XAI explanations are stored in audit logs only */}
          
          {/* Ticket Info */}
          <div className="bg-white shadow rounded-lg">
            <div className="px-4 py-5 sm:px-6">
              <h3 className="text-lg leading-6 font-medium text-gray-900">Ticket Information</h3>
              <dl className="mt-4 space-y-3">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Requester</dt>
                  <dd className="text-sm text-gray-900">User #{ticket.requester_id}</dd>
                </div>
                {ticket.assigned_agent_id && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Assigned Agent</dt>
                    <dd className="text-sm text-gray-900">Agent #{ticket.assigned_agent_id}</dd>
                  </div>
                )}
                <div>
                  <dt className="text-sm font-medium text-gray-500">Created</dt>
                  <dd className="text-sm text-gray-900">{new Date(ticket.created_at).toLocaleString()}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Last Updated</dt>
                  <dd className="text-sm text-gray-900">{new Date(ticket.updated_at).toLocaleString()}</dd>
                </div>
                {ticket.due_date && (
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Due Date</dt>
                    <dd className="text-sm text-gray-900">{new Date(ticket.due_date).toLocaleString()}</dd>
                  </div>
                )}
              </dl>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TicketDetail;
