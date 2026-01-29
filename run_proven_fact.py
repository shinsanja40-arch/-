#!/usr/bin/env python3
"""
Command-line interface for the Proven Fact-Based Algorithm
"""

import argparse
import sys
import json
from proven_fact_system import ProvenFactSystem


# Predefined simulation templates
SIMULATION_TEMPLATES = {
    "earth_rotation": {
        "proven_fact": "The Earth rotates on its axis once every 24 hours, causing day and night cycles.",
        "topic": "Earth's Rotation and the Day-Night Cycle",
        "evidence_stages": [
            [
                "The Sun appears to rise in the east and set in the west daily",
                "Stars appear to move across the night sky in circular patterns",
                "Shadows change length and direction throughout the day"
            ],
            [
                "Ancient Greek astronomers measured stellar positions",
                "Sundials have been used for thousands of years to track time",
                "Different locations experience noon at different times"
            ],
            [
                "Foucault pendulum demonstrates Earth's rotation (1851)",
                "Satellites in geostationary orbit remain fixed above one point",
                "Coriolis effect influences weather patterns and ocean currents",
                "High-precision atomic clocks confirm 24-hour rotation period"
            ],
            [
                "Photos from space show Earth rotating",
                "GPS satellites account for Earth's rotation in positioning calculations",
                "International Space Station completes orbit while Earth rotates beneath"
            ]
        ]
    },
    "earth_sphericity": {
        "proven_fact": "The Earth is a sphere (oblate spheroid) with a circumference of approximately 40,075 km at the equator.",
        "topic": "The Spherical Shape of Earth",
        "evidence_stages": [
            [
                "Ships disappear over the horizon hull-first, mast-last",
                "The shadow of Earth on the Moon during lunar eclipses is always circular",
                "Different stars are visible from different latitudes"
            ],
            [
                "Eratosthenes measured Earth's circumference (~250 BCE) using shadows at different locations",
                "Travelers going south see southern stars rise higher in the sky",
                "Altitude of Polaris changes with latitude"
            ],
            [
                "Ferdinand Magellan's crew completed first circumnavigation (1519-1522)",
                "Time zones exist because different longitudes face the sun at different times",
                "Gravity measurements show consistent spherical distribution"
            ],
            [
                "Satellite photos from space show Earth as a sphere",
                "GPS system requires spherical Earth model for accurate positioning",
                "Astronauts have directly observed Earth's curvature from space"
            ]
        ]
    },
    "evolution": {
        "proven_fact": "Species evolve over time through natural selection, as evidenced by fossil records, DNA analysis, and observable adaptation.",
        "topic": "Biological Evolution and Natural Selection",
        "evidence_stages": [
            [
                "Fossils show progression of species over geological time",
                "Different species share similar anatomical structures (homology)",
                "Selective breeding produces visible changes in domesticated species"
            ],
            [
                "Darwin's finches show beak adaptations to different food sources",
                "Vestigial structures (appendix, tailbone) suggest evolutionary history",
                "Embryological similarities across species indicate common ancestry"
            ],
            [
                "DNA sequencing reveals genetic relationships between species",
                "Antibiotic resistance in bacteria demonstrates natural selection in real-time",
                "Molecular clocks date species divergence through mutation rates"
            ],
            [
                "CRISPR reveals shared genetic mechanisms across all life",
                "Observed speciation events in controlled environments",
                "Genetic studies confirm human-ape common ancestry (~6-7 million years ago)"
            ]
        ]
    },
    "climate_change": {
        "proven_fact": "Global average temperatures have increased approximately 1.1°C since pre-industrial times, primarily due to human greenhouse gas emissions.",
        "topic": "Anthropogenic Climate Change",
        "evidence_stages": [
            [
                "Global temperature measurements show consistent warming trend since 1880",
                "Arctic sea ice extent has decreased significantly",
                "Glaciers worldwide are retreating"
            ],
            [
                "Ice core data shows CO2 levels are highest in 800,000 years",
                "Isotopic analysis confirms CO2 increase is from fossil fuels",
                "Ocean heat content has increased measurably"
            ],
            [
                "Atmospheric CO2 has risen from 280 ppm (pre-industrial) to 420 ppm (current)",
                "Satellite measurements confirm decreasing outgoing infrared radiation",
                "Sea level has risen ~20 cm since 1900, accelerating in recent decades"
            ],
            [
                "Climate models accurately predicted warming patterns",
                "Attribution studies show natural factors alone cannot explain warming",
                "Extreme weather events show patterns consistent with climate predictions"
            ]
        ]
    }
}


def main():
    parser = argparse.ArgumentParser(
        description='Run the Proven Fact-Based Algorithm for AI Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use predefined template
  python run_proven_fact.py --template earth_rotation --sessions 12

  # Custom simulation from JSON config
  python run_proven_fact.py --config my_simulation.json --sessions 15

  # With specific API provider
  python run_proven_fact.py --template earth_sphericity --provider openai --sessions 15

Available templates: earth_rotation, earth_sphericity, evolution, climate_change
        """
    )
    
    parser.add_argument(
        '--template',
        type=str,
        choices=list(SIMULATION_TEMPLATES.keys()),
        help='Use a predefined simulation template'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='JSON file with custom simulation configuration'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['anthropic', 'openai'],
        default='anthropic',
        help='LLM provider to use (default: anthropic)'
    )
    
    parser.add_argument(
        '--sessions',
        type=int,
        default=12,
        help='Number of learning sessions (default: 12)'
    )
    
    parser.add_argument(
        '--professors',
        type=int,
        default=2,
        help='Number of professor agents (default: 2)'
    )
    
    parser.add_argument(
        '--referees',
        type=int,
        default=2,
        help='Number of referee agents (default: 2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for results (default: auto-generated)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='API key (if not set in environment)'
    )
    
    args = parser.parse_args()
    
    # Load simulation config
    if args.template:
        config = SIMULATION_TEMPLATES[args.template]
        output_file = args.output or f"{args.template}_results.json"
    elif args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        output_file = args.output or "custom_simulation_results.json"
    else:
        print("Error: Must specify either --template or --config")
        sys.exit(1)
    
    # Validate config
    required_keys = ['proven_fact', 'topic', 'evidence_stages']
    for key in required_keys:
        if key not in config:
            print(f"Error: Config missing required key: {key}")
            sys.exit(1)
    
    # Initialize system
    try:
        print(f"\nInitializing Proven Fact System...")
        print(f"Provider: {args.provider}")
        print(f"Professors: {args.professors}")
        print(f"Referees: {args.referees}")
        print(f"Sessions: {args.sessions}\n")
        
        system = ProvenFactSystem(
            api_provider=args.provider,
            api_key=args.api_key,
            num_professors=args.professors,
            num_referees=args.referees
        )
        
        # Run simulation
        metrics = system.run_learning_simulation(
            proven_fact=config['proven_fact'],
            topic=config['topic'],
            evidence_stages=config['evidence_stages'],
            total_sessions=args.sessions,
            output_file=output_file
        )
        
        print(f"\n{'='*70}")
        print("SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Results saved to: {output_file}")
        print(f"\nTo analyze results, run:")
        print(f"  python analyze_proven_fact.py {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
