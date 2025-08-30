#!/usr/bin/env python3
"""
Knowledge Graph Management CLI

This script provides command-line utilities for managing the knowledge graph:
- Building/rebuilding the graph from policy documents
- Visualizing the graph structure
- Adding manual relationships
- Exporting the graph to various formats
- Analyzing graph statistics and coverage
"""

import os
import sys
import argparse
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from app.core.database import SessionLocal
from app.models.models import KnowledgeGraphConcept, PolicyDocument
from app.services.knowledge_graph_builder import PolicyKnowledgeGraphBuilder
from app.services.knowledge_graph_query import KnowledgeGraphQueryService


def rebuild_graph(args):
    """Rebuild the knowledge graph from policy documents"""
    db = SessionLocal()
    try:
        kg_builder = PolicyKnowledgeGraphBuilder()
        
        print("🕸️ Building knowledge graph from policy documents...")
        start_time = time.time()
        
        result = kg_builder.rebuild_graph(db)
        
        end_time = time.time()
        print(f"✅ Knowledge graph built successfully in {end_time - start_time:.2f} seconds")
        print(f"  - Concepts created: {result['concepts_created']}")
        print(f"  - Extractions found: {result['total_extractions']}")
        print(f"  - Relationships created: {result['total_relationships']}")
        
        # Print stats
        stats = kg_builder.get_graph_statistics(db)
        print("\n📊 Graph Statistics:")
        print(f"  - Total concepts: {stats['total_concepts']}")
        print(f"  - Total relationships: {stats['total_relationships']}")
        print(f"  - Avg relationships per concept: {stats['avg_relationships_per_concept']:.2f}")
        
        if 'relationship_types' in stats:
            print("\n🔗 Relationship Types:")
            for rel_type, count in stats['relationship_types'].items():
                print(f"  - {rel_type}: {count}")
        
    finally:
        db.close()


def show_stats(args):
    """Show statistics about the knowledge graph"""
    db = SessionLocal()
    try:
        kg_builder = PolicyKnowledgeGraphBuilder()
        kg_query = KnowledgeGraphQueryService()
        
        stats = kg_builder.get_graph_statistics(db)
        print("\n📊 Graph Statistics:")
        print(f"  - Total concepts: {stats['total_concepts']}")
        print(f"  - Total relationships: {stats['total_relationships']}")
        print(f"  - Total extractions: {stats['total_extractions']}")
        print(f"  - Avg relationships per concept: {stats['avg_relationships_per_concept']:.2f}")
        
        if 'relationship_types' in stats:
            print("\n🔗 Relationship Types:")
            for rel_type, count in stats['relationship_types'].items():
                print(f"  - {rel_type}: {count}")
        
        # Show most connected concepts
        print("\n🌟 Most Connected Concepts:")
        connected_concepts = kg_query.get_most_connected_concepts(db, top_k=5)
        for i, concept in enumerate(connected_concepts):
            print(f"  {i+1}. {concept['concept_name']} ({concept['concept_type']}) - Centrality: {concept['combined_centrality']:.3f}")
            
    finally:
        db.close()


def list_concepts(args):
    """List all concepts in the knowledge graph"""
    db = SessionLocal()
    try:
        concept_type = args.type
        query = db.query(KnowledgeGraphConcept)
        
        if concept_type:
            query = query.filter(KnowledgeGraphConcept.concept_type == concept_type)
            
        concepts = query.all()
        
        print(f"\n📋 Knowledge Graph Concepts ({len(concepts)}):")
        for i, concept in enumerate(concepts):
            chunk_count = len(concept.policy_chunks) if concept.policy_chunks else 0
            print(f"  {i+1}. {concept.name} (Type: {concept.concept_type}, Chunks: {chunk_count})")
            
            if args.verbose and concept.aliases:
                print(f"     Aliases: {', '.join(concept.aliases)}")
                
    finally:
        db.close()


def add_relationship(args):
    """Add a manual relationship between two concepts"""
    db = SessionLocal()
    try:
        kg_builder = PolicyKnowledgeGraphBuilder()
        
        source = args.source
        target = args.target
        rel_type = args.type
        weight = args.weight
        description = args.description
        
        if not all([source, target, rel_type]):
            print("❌ Error: Missing required parameters (source, target, type)")
            return
        
        print(f"🔗 Adding relationship: {source} --[{rel_type}]--> {target}")
        
        success = kg_builder.add_manual_relationship(
            db=db,
            source_concept=source,
            target_concept=target,
            relationship_type=rel_type,
            weight=weight,
            description=description
        )
        
        if success:
            print("✅ Relationship added successfully")
        else:
            print("❌ Failed to add relationship. Check that both concepts exist.")
            
    finally:
        db.close()


def visualize_graph(args):
    """Visualize the knowledge graph using NetworkX and matplotlib"""
    db = SessionLocal()
    try:
        kg_builder = PolicyKnowledgeGraphBuilder()
        
        # Export graph to NetworkX
        G = kg_builder.export_graph_to_networkx(db)
        
        if not G.nodes():
            print("❌ Error: Graph is empty. Build the graph first.")
            return
        
        # Limit to maximum number of nodes for visualization
        max_nodes = args.max_nodes
        if len(G.nodes()) > max_nodes:
            print(f"⚠️  Warning: Graph has {len(G.nodes())} nodes. Limiting visualization to {max_nodes} most connected nodes.")
            # Get most connected nodes by degree
            sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
            node_ids = [n[0] for n in sorted_nodes]
            G = G.subgraph(node_ids)
        
        # Get node labels
        node_labels = {}
        for node_id in G.nodes():
            concept = db.query(KnowledgeGraphConcept).get(node_id)
            if concept:
                node_labels[node_id] = concept.name
        
        # Node colors based on concept type
        color_map = {
            "technology": "skyblue",
            "security": "salmon",
            "policy": "lightgreen",
            "procedure": "khaki",
            "requirement": "plum"
        }
        
        node_colors = []
        for node_id in G.nodes():
            concept = db.query(KnowledgeGraphConcept).get(node_id)
            if concept:
                node_colors.append(color_map.get(concept.concept_type, "lightgray"))
        
        # Set up the plot
        plt.figure(figsize=(15, 10))
        
        # Choose layout
        if args.layout == "spring":
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        elif args.layout == "circular":
            pos = nx.circular_layout(G)
        elif args.layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:  # Default to spring
            pos = nx.spring_layout(G)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        # Draw edges with different colors based on relationship type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            if data.get('relationship_type') == 'requires':
                edge_colors.append('red')
            elif data.get('relationship_type') == 'depends_on':
                edge_colors.append('blue')
            elif data.get('relationship_type') == 'overrides':
                edge_colors.append('green')
            else:
                edge_colors.append('gray')
                
        nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors, alpha=0.7)
        
        # Add a title
        plt.title("Knowledge Graph Visualization", fontsize=16)
        plt.axis('off')
        
        # Save or show the graph
        if args.output:
            plt.savefig(args.output, format='png', dpi=300, bbox_inches='tight')
            print(f"✅ Graph visualization saved to {args.output}")
        else:
            plt.show()
            
    finally:
        db.close()


def export_graph(args):
    """Export the knowledge graph to various formats"""
    db = SessionLocal()
    try:
        kg_builder = PolicyKnowledgeGraphBuilder()
        
        # Export graph to NetworkX
        G = kg_builder.export_graph_to_networkx(db)
        
        if not G.nodes():
            print("❌ Error: Graph is empty. Build the graph first.")
            return
        
        output_format = args.format.lower()
        output_file = args.output
        
        # Export to specified format
        if output_format == "gml":
            nx.write_gml(G, output_file)
        elif output_format == "graphml":
            nx.write_graphml(G, output_file)
        elif output_format == "adjlist":
            nx.write_adjlist(G, output_file)
        elif output_format == "json":
            # Convert to JSON manually
            data = {
                "nodes": [],
                "edges": []
            }
            
            # Add nodes with attributes
            for node_id, attrs in G.nodes(data=True):
                concept = db.query(KnowledgeGraphConcept).get(node_id)
                if concept:
                    data["nodes"].append({
                        "id": node_id,
                        "name": concept.name,
                        "type": concept.concept_type,
                        "importance": concept.importance_score,
                        "aliases": concept.aliases,
                        **attrs
                    })
            
            # Add edges with attributes
            for source, target, attrs in G.edges(data=True):
                data["edges"].append({
                    "source": source,
                    "target": target,
                    "relationship": attrs.get("relationship_type", "related"),
                    "weight": attrs.get("weight", 1.0),
                    **attrs
                })
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            print(f"❌ Unsupported format: {output_format}")
            return
        
        print(f"✅ Graph exported to {output_file} in {output_format} format")
        
    finally:
        db.close()


def search_policies(args):
    """Test KG-Enhanced RAG with a query"""
    db = SessionLocal()
    try:
        from app.services.policy_retriever import PolicyRetriever
        
        query = args.query
        k = args.top_k
        enable_kg = not args.disable_kg
        max_hops = args.max_hops
        
        policy_retriever = PolicyRetriever()
        
        print(f"🔍 Searching policies with query: '{query}'")
        print(f"Settings: enable_kg={enable_kg}, max_hops={max_hops}, k={k}")
        
        start_time = time.time()
        enhanced_citations, graph_hops, metadata = policy_retriever.kg_enhanced_retrieve(
            query=query,
            k=k,
            enable_kg=enable_kg,
            max_graph_hops=max_hops,
            db=db
        )
        end_time = time.time()
        
        # Generate explanation
        explanation = policy_retriever.explain_retrieval_reasoning(
            query=query,
            enhanced_citations=enhanced_citations,
            graph_hops=graph_hops,
            metadata=metadata
        )
        
        print(f"\n⏱️  Retrieval time: {end_time - start_time:.2f} seconds")
        print(f"📄 Found {len(enhanced_citations)} relevant policy citations")
        
        # Show explanation
        print("\n" + "=" * 80)
        print(explanation)
        print("=" * 80)
        
        # Show top results
        print("\n📝 Top Policy Chunks:")
        for i, citation in enumerate(enhanced_citations[:5]):  # Show top 5
            print(f"\n{i+1}. {citation.document_title} (Score: {citation.combined_score:.3f})")
            print(f"   Semantic: {citation.semantic_score:.3f}, Graph: {citation.graph_boost_score:.3f}")
            if citation.graph_path:
                print(f"   Found via graph: {len(citation.graph_path)} connections")
            print(f"   Content: {citation.chunk_content[:150]}...")
        
    finally:
        db.close()


def get_neighborhood(args):
    """Get concept neighborhood"""
    db = SessionLocal()
    try:
        kg_query = KnowledgeGraphQueryService()
        
        concept_name = args.concept
        radius = args.radius
        
        print(f"🔍 Getting neighborhood for concept: '{concept_name}' (radius={radius})")
        
        neighborhood = kg_query.get_concept_neighborhood(db, concept_name, radius)
        
        if not neighborhood:
            print(f"❌ Concept '{concept_name}' not found or has no neighbors")
            return
        
        for depth, neighbors in neighborhood.items():
            print(f"\n📡 {depth.replace('_', ' ').title()}:")
            if neighbors:
                for i, neighbor in enumerate(neighbors):
                    print(f"  {i+1}. {neighbor}")
            else:
                print("  No concepts at this depth")
        
    finally:
        db.close()


def analyze_policy_coverage(args):
    """Analyze how well policies are covered in the knowledge graph"""
    db = SessionLocal()
    try:
        # Get all policy documents
        policies = db.query(PolicyDocument).all()
        
        if not policies:
            print("❌ No policy documents found")
            return
        
        print(f"📊 Analyzing policy coverage for {len(policies)} documents\n")
        
        # Get all concepts
        from sqlalchemy import func
        from app.models.models import PolicyConceptExtraction
        
        # For each policy, get coverage stats
        for policy in policies:
            chunks = policy.chunks
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                print(f"Policy '{policy.title}': No chunks")
                continue
            
            # Count chunks with concept extractions
            chunks_with_concepts = db.query(func.count(PolicyConceptExtraction.chunk_id.distinct()))\
                .filter(PolicyConceptExtraction.chunk_id.in_([c.id for c in chunks]))\
                .scalar()
            
            # Calculate coverage percentage
            coverage_pct = (chunks_with_concepts / total_chunks) * 100
            
            # Get top concepts for this policy
            top_concepts = db.query(
                KnowledgeGraphConcept.name,
                func.count(PolicyConceptExtraction.id).label('count')
            ).join(
                PolicyConceptExtraction, 
                KnowledgeGraphConcept.id == PolicyConceptExtraction.concept_id
            ).filter(
                PolicyConceptExtraction.chunk_id.in_([c.id for c in chunks])
            ).group_by(
                KnowledgeGraphConcept.name
            ).order_by(
                func.count(PolicyConceptExtraction.id).desc()
            ).limit(5).all()
            
            # Print results
            print(f"Policy: {policy.title}")
            print(f"  - Total chunks: {total_chunks}")
            print(f"  - Chunks with concepts: {chunks_with_concepts} ({coverage_pct:.1f}%)")
            
            if top_concepts:
                print("  - Top concepts:")
                for concept, count in top_concepts:
                    print(f"    • {concept}: {count} mentions")
            print()
        
    finally:
        db.close()


def setup_parser():
    """Set up command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rebuild the knowledge graph
  python manage_kg.py rebuild
  
  # Show graph statistics
  python manage_kg.py stats
  
  # List all concepts (with verbose output)
  python manage_kg.py list-concepts -v
  
  # List concepts of a specific type
  python manage_kg.py list-concepts -t security
  
  # Add a relationship between concepts
  python manage_kg.py add-relationship -s "VPN" -t "MFA" -t "requires" -w 0.9
  
  # Visualize the graph
  python manage_kg.py visualize -l spring -o graph.png
  
  # Export the graph to JSON
  python manage_kg.py export -f json -o graph.json
  
  # Test KG-Enhanced RAG search
  python manage_kg.py search -q "VPN access issues"
  
  # Get concept neighborhood
  python manage_kg.py neighborhood -c "VPN" -r 2
  
  # Analyze policy coverage
  python manage_kg.py analyze-coverage
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Rebuild command
    rebuild_parser = subparsers.add_parser('rebuild', help='Rebuild the knowledge graph')
    rebuild_parser.set_defaults(func=rebuild_graph)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show graph statistics')
    stats_parser.set_defaults(func=show_stats)
    
    # List concepts command
    list_parser = subparsers.add_parser('list-concepts', help='List all concepts')
    list_parser.add_argument('-t', '--type', help='Filter by concept type')
    list_parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')
    list_parser.set_defaults(func=list_concepts)
    
    # Add relationship command
    rel_parser = subparsers.add_parser('add-relationship', help='Add a relationship between concepts')
    rel_parser.add_argument('-s', '--source', required=True, help='Source concept name')
    rel_parser.add_argument('-t', '--target', required=True, help='Target concept name')
    rel_parser.add_argument('-y', '--type', required=True, help='Relationship type')
    rel_parser.add_argument('-w', '--weight', type=float, default=1.0, help='Relationship weight')
    rel_parser.add_argument('-d', '--description', help='Relationship description')
    rel_parser.set_defaults(func=add_relationship)
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize the graph')
    viz_parser.add_argument('-l', '--layout', choices=['spring', 'circular', 'kamada_kawai'],
                            default='spring', help='Graph layout algorithm')
    viz_parser.add_argument('-o', '--output', help='Output file path (PNG)')
    viz_parser.add_argument('-m', '--max-nodes', type=int, default=50, 
                            help='Maximum number of nodes to visualize')
    viz_parser.set_defaults(func=visualize_graph)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export the graph')
    export_parser.add_argument('-f', '--format', choices=['gml', 'graphml', 'adjlist', 'json'],
                              required=True, help='Export format')
    export_parser.add_argument('-o', '--output', required=True, help='Output file path')
    export_parser.set_defaults(func=export_graph)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Test KG-Enhanced RAG with a query')
    search_parser.add_argument('-q', '--query', required=True, help='Search query')
    search_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--disable-kg', action='store_true', help='Disable knowledge graph enhancement')
    search_parser.add_argument('-m', '--max-hops', type=int, default=2, help='Maximum graph hops')
    search_parser.set_defaults(func=search_policies)
    
    # Neighborhood command
    neighborhood_parser = subparsers.add_parser('neighborhood', help='Get concept neighborhood')
    neighborhood_parser.add_argument('-c', '--concept', required=True, help='Concept name')
    neighborhood_parser.add_argument('-r', '--radius', type=int, default=1, help='Neighborhood radius')
    neighborhood_parser.set_defaults(func=get_neighborhood)
    
    # Analyze coverage command
    coverage_parser = subparsers.add_parser('analyze-coverage', help='Analyze policy coverage')
    coverage_parser.set_defaults(func=analyze_policy_coverage)
    
    return parser


def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
