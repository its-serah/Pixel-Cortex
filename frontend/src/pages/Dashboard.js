import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from 'react-query';
import { Ticket, Plus, Brain, Shield } from 'lucide-react';
import api from '../utils/api';

function Dashboard() {
  const { data: tickets } = useQuery(
    'recent-tickets',
    () => api.get('/api/tickets?limit=5').then(res => res.data),
    { enabled: true }
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Ticket className="h-6 w-6 text-gray-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Total Tickets</dt>
                  <dd className="text-lg font-medium text-gray-900">{tickets?.length || 0}</dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Brain className="h-6 w-6 text-primary-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">AI Processed</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {tickets?.filter(t => t.triage_confidence).length || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Shield className="h-6 w-6 text-green-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Avg Confidence</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {tickets?.filter(t => t.triage_confidence).length > 0
                      ? Math.round(
                          tickets.filter(t => t.triage_confidence)
                            .reduce((sum, t) => sum + t.triage_confidence, 0) /
                          tickets.filter(t => t.triage_confidence).length * 100
                        ) + '%'
                      : 'N/A'}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center justify-center">
              <Link
                to="/tickets/new"
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700"
              >
                <Plus className="h-4 w-4 mr-2" />
                New Ticket
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Tickets */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:px-6 border-b border-gray-200">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Recent Tickets</h3>
        </div>
        <ul className="divide-y divide-gray-200">
          {tickets?.slice(0, 5).map((ticket) => (
            <li key={ticket.id}>
              <Link to={`/tickets/${ticket.id}`} className="block hover:bg-gray-50">
                <div className="px-4 py-4 flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      #{ticket.id} {ticket.title}
                    </p>
                    <div className="mt-2 flex items-center text-sm text-gray-500 space-x-4">
                      <span className="capitalize">{ticket.category}</span>
                      <span>•</span>
                      <span className="capitalize">{ticket.priority}</span>
                      <span>•</span>
                      <span>{new Date(ticket.created_at).toLocaleDateString()}</span>
                      {ticket.triage_confidence && (
                        <>
                          <span>•</span>
                          <span className="text-primary-600">
                            AI: {Math.round(ticket.triage_confidence * 100)}%
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default Dashboard;
