"""
Astrological Entity Dictionary
Comprehensive collection of astrological terms and their variations
"""

from typing import Dict, List, Set
import re


class AstroEntityDictionary:
    """Dictionary of all astrological entities and their variations"""
    
    def __init__(self):
        """Initialize the dictionary with all astrological entities"""
        
        # Planets (Grahas)
        self.planets = {
            'Sun': ['Sun', 'Surya', 'Ravi', 'Solar'],
            'Moon': ['Moon', 'Chandra', 'Soma', 'Lunar'],
            'Mars': ['Mars', 'Mangal', 'Kuja', 'Angaraka'],
            'Mercury': ['Mercury', 'Budh', 'Budha'],
            'Jupiter': ['Jupiter', 'Guru', 'Brihaspati', 'Jove'],
            'Venus': ['Venus', 'Shukra', 'Sukra'],
            'Saturn': ['Saturn', 'Shani', 'Sani'],
            'Rahu': ['Rahu', 'North Node', 'Dragon\'s Head'],
            'Ketu': ['Ketu', 'South Node', 'Dragon\'s Tail']
        }
        
        # Houses (Bhavas)
        self.houses = {
            '1st': ['1st house', 'first house', 'ascendant', 'lagna', 'rising sign'],
            '2nd': ['2nd house', 'second house'],
            '3rd': ['3rd house', 'third house'],
            '4th': ['4th house', 'fourth house'],
            '5th': ['5th house', 'fifth house'],
            '6th': ['6th house', 'sixth house'],
            '7th': ['7th house', 'seventh house'],
            '8th': ['8th house', 'eighth house'],
            '9th': ['9th house', 'ninth house'],
            '10th': ['10th house', 'tenth house', 'midheaven', 'MC'],
            '11th': ['11th house', 'eleventh house'],
            '12th': ['12th house', 'twelfth house']
        }
        
        # Zodiac Signs (Rashis)
        self.signs = {
            'Aries': ['Aries', 'Mesha', 'Mesh', 'Ram'],
            'Taurus': ['Taurus', 'Vrishabha', 'Vrishaba', 'Bull'],
            'Gemini': ['Gemini', 'Mithuna', 'Twins'],
            'Cancer': ['Cancer', 'Karka', 'Karkata', 'Crab'],
            'Leo': ['Leo', 'Simha', 'Lion'],
            'Virgo': ['Virgo', 'Kanya', 'Virgin'],
            'Libra': ['Libra', 'Tula', 'Balance'],
            'Scorpio': ['Scorpio', 'Vrishchika', 'Scorpion'],
            'Sagittarius': ['Sagittarius', 'Dhanu', 'Dhanus', 'Archer'],
            'Capricorn': ['Capricorn', 'Makara', 'Makar', 'Goat'],
            'Aquarius': ['Aquarius', 'Kumbha', 'Water Bearer'],
            'Pisces': ['Pisces', 'Meena', 'Fish']
        }
        
        # Nakshatras (Lunar Mansions)
        self.nakshatras = {
            'Ashwini': ['Ashwini', 'Aswini'],
            'Bharani': ['Bharani'],
            'Krittika': ['Krittika', 'Kritika'],
            'Rohini': ['Rohini'],
            'Mrigashira': ['Mrigashira', 'Mrigasira'],
            'Ardra': ['Ardra', 'Arudra'],
            'Punarvasu': ['Punarvasu'],
            'Pushya': ['Pushya', 'Pushyami'],
            'Ashlesha': ['Ashlesha', 'Aslesha'],
            'Magha': ['Magha'],
            'Purva Phalguni': ['Purva Phalguni', 'Purvaphalguni'],
            'Uttara Phalguni': ['Uttara Phalguni', 'Uttaraphalguni'],
            'Hasta': ['Hasta'],
            'Chitra': ['Chitra'],
            'Swati': ['Swati', 'Svati'],
            'Vishakha': ['Vishakha', 'Visakha'],
            'Anuradha': ['Anuradha'],
            'Jyeshtha': ['Jyeshtha', 'Jyestha'],
            'Mula': ['Mula', 'Moola'],
            'Purva Ashadha': ['Purva Ashadha', 'Purvashada'],
            'Uttara Ashadha': ['Uttara Ashadha', 'Uttarashada'],
            'Shravana': ['Shravana', 'Sravana'],
            'Dhanishta': ['Dhanishta', 'Dhanistha'],
            'Shatabhisha': ['Shatabhisha', 'Satabhisha'],
            'Purva Bhadrapada': ['Purva Bhadrapada', 'Purvabhadra'],
            'Uttara Bhadrapada': ['Uttara Bhadrapada', 'Uttarabhadra'],
            'Revati': ['Revati']
        }
        
        # Dashas (Planetary Periods)
        self.dashas = {
            'Mahadasha': ['Mahadasha', 'Maha Dasha', 'major period'],
            'Antardasha': ['Antardasha', 'Antar Dasha', 'sub period'],
            'Pratyantardasha': ['Pratyantardasha', 'Pratyantar Dasha'],
            'Sun Dasha': ['Sun Dasha', 'Sun Mahadasha', 'Surya Dasha'],
            'Moon Dasha': ['Moon Dasha', 'Moon Mahadasha', 'Chandra Dasha'],
            'Mars Dasha': ['Mars Dasha', 'Mars Mahadasha', 'Mangal Dasha'],
            'Mercury Dasha': ['Mercury Dasha', 'Mercury Mahadasha', 'Budh Dasha'],
            'Jupiter Dasha': ['Jupiter Dasha', 'Jupiter Mahadasha', 'Guru Dasha'],
            'Venus Dasha': ['Venus Dasha', 'Venus Mahadasha', 'Shukra Dasha'],
            'Saturn Dasha': ['Saturn Dasha', 'Saturn Mahadasha', 'Shani Dasha'],
            'Rahu Dasha': ['Rahu Dasha', 'Rahu Mahadasha'],
            'Ketu Dasha': ['Ketu Dasha', 'Ketu Mahadasha']
        }
        
        # Aspects
        self.aspects = {
            'Conjunction': ['conjunction', 'conjunct', 'together with', 'combined with'],
            'Opposition': ['opposition', 'opposite', 'opposing'],
            'Trine': ['trine', 'trinal aspect', '120 degrees'],
            'Square': ['square', 'squared', '90 degrees'],
            'Sextile': ['sextile', '60 degrees'],
            'Aspect': ['aspect', 'aspects', 'aspected by', 'aspecting']
        }
        
        # Yogas (Planetary Combinations)
        self.yogas = {
            'Raj Yoga': ['Raj Yoga', 'Raja Yoga', 'Royal Yoga'],
            'Dhana Yoga': ['Dhana Yoga', 'Wealth Yoga'],
            'Gaja Kesari': ['Gaja Kesari', 'Gajakesari', 'Gajkesari'],
            'Neecha Bhanga': ['Neecha Bhanga', 'Neecha Bhang', 'cancellation of debilitation'],
            'Pancha Mahapurusha': ['Pancha Mahapurusha', 'Five Great Men'],
            'Hamsa Yoga': ['Hamsa Yoga'],
            'Malavya Yoga': ['Malavya Yoga'],
            'Ruchaka Yoga': ['Ruchaka Yoga'],
            'Bhadra Yoga': ['Bhadra Yoga'],
            'Shasha Yoga': ['Shasha Yoga'],
            'Viparita Raja Yoga': ['Viparita Raja Yoga'],
            'Budhaditya Yoga': ['Budhaditya Yoga', 'Sun Mercury Yoga'],
            'Chandra Mangala Yoga': ['Chandra Mangala Yoga', 'Moon Mars Yoga']
        }
        
        # Dignities
        self.dignities = {
            'Exalted': ['exalted', 'exaltation', 'uccha'],
            'Debilitated': ['debilitated', 'debilitation', 'neecha', 'fallen'],
            'Own Sign': ['own sign', 'swakshetra', 'swa-kshetra'],
            'Moolatrikona': ['moolatrikona', 'moola-trikona'],
            'Friend': ['friendly', 'friend sign'],
            'Enemy': ['enemy', 'enemy sign'],
            'Neutral': ['neutral', 'neutral sign']
        }
        
        # House Significations (Keywords)
        self.house_significations = {
            '1st': ['self', 'personality', 'appearance', 'body', 'identity'],
            '2nd': ['wealth', 'family', 'speech', 'food', 'assets'],
            '3rd': ['siblings', 'courage', 'communication', 'short journeys'],
            '4th': ['mother', 'home', 'property', 'vehicles', 'happiness'],
            '5th': ['children', 'creativity', 'education', 'romance', 'speculation'],
            '6th': ['enemies', 'disease', 'service', 'debts', 'obstacles'],
            '7th': ['spouse', 'marriage', 'partnerships', 'business'],
            '8th': ['longevity', 'death', 'inheritance', 'occult', 'transformation'],
            '9th': ['father', 'luck', 'dharma', 'long journeys', 'higher education'],
            '10th': ['career', 'profession', 'status', 'authority', 'reputation'],
            '11th': ['gains', 'income', 'friends', 'desires', 'elder siblings'],
            '12th': ['loss', 'expenses', 'foreign', 'spirituality', 'moksha']
        }
        
        # Relationship Keywords
        self.relationships = {
            'placed_in': ['in', 'placed in', 'located in', 'posited in', 'situated in'],
            'aspected_by': ['aspected by', 'aspects from', 'receiving aspect'],
            'rules': ['rules', 'lord of', 'owns', 'ruling'],
            'signifies': ['signifies', 'represents', 'indicates', 'means'],
            'creates': ['creates', 'forms', 'produces', 'makes'],
            'influences': ['influences', 'affects', 'impacts'],
            'conjoined_with': ['conjunct', 'with', 'together with', 'combined with'],
            'transiting': ['transiting', 'transit', 'moving through']
        }
    
    def get_all_entities(self) -> Dict[str, Dict]:
        """Get all entity categories"""
        return {
            'planets': self.planets,
            'houses': self.houses,
            'signs': self.signs,
            'nakshatras': self.nakshatras,
            'dashas': self.dashas,
            'aspects': self.aspects,
            'yogas': self.yogas,
            'dignities': self.dignities
        }
    
    def get_entity_variations(self, entity: str, category: str = None) -> List[str]:
        """
        Get all variations of an entity
        
        Args:
            entity: Entity name
            category: Optional category to search in
            
        Returns:
            List of variations
        """
        if category:
            category_dict = getattr(self, category, {})
            return category_dict.get(entity, [entity])
        
        # Search all categories
        for cat_dict in [self.planets, self.houses, self.signs, self.nakshatras, 
                         self.dashas, self.aspects, self.yogas, self.dignities]:
            if entity in cat_dict:
                return cat_dict[entity]
        
        return [entity]
    
    def get_regex_patterns(self) -> Dict[str, List[str]]:
        """
        Get regex patterns for all entities
        
        Returns:
            Dictionary of category -> list of regex patterns
        """
        patterns = {}
        
        for category, entities in self.get_all_entities().items():
            category_patterns = []
            for entity, variations in entities.items():
                # Create regex pattern that matches any variation
                pattern = '|'.join(re.escape(v) for v in variations)
                category_patterns.append(f'({pattern})')
            patterns[category] = category_patterns
        
        return patterns
    
    def find_entity_type(self, text: str) -> tuple:
        """
        Find which entity type a text belongs to
        
        Args:
            text: Text to check
            
        Returns:
            Tuple of (category, entity_name) or (None, None)
        """
        text_lower = text.lower()
        
        for category, entities in self.get_all_entities().items():
            for entity_name, variations in entities.items():
                for variation in variations:
                    if variation.lower() == text_lower:
                        return (category, entity_name)
        
        return (None, None)
    
    def get_house_keywords(self, house: str) -> List[str]:
        """Get signification keywords for a house"""
        return self.house_significations.get(house, [])
    
    def get_relationship_patterns(self) -> Dict[str, List[str]]:
        """Get relationship keyword patterns"""
        return self.relationships


# Example usage
if __name__ == "__main__":
    dictionary = AstroEntityDictionary()
    
    print("=== Astrological Entity Dictionary ===\n")
    
    # Show planets
    print("Planets:")
    for planet, variations in dictionary.planets.items():
        print(f"  {planet}: {', '.join(variations)}")
    
    # Show houses
    print("\nHouses (sample):")
    for house in ['1st', '7th', '10th']:
        variations = dictionary.houses[house]
        print(f"  {house}: {', '.join(variations)}")
    
    # Test entity lookup
    print("\n=== Entity Lookup Tests ===")
    test_entities = ['Mars', 'Mangal', '10th house', 'Guru', 'Gaja Kesari']
    for entity in test_entities:
        category, name = dictionary.find_entity_type(entity)
        print(f"'{entity}' -> Category: {category}, Name: {name}")
    
    # Show relationship keywords
    print("\n=== Relationship Keywords ===")
    for rel_type, keywords in dictionary.relationships.items():
        print(f"  {rel_type}: {', '.join(keywords[:3])}...")
    
    # Total counts
    all_entities = dictionary.get_all_entities()
    total = sum(len(entities) for entities in all_entities.values())
    print(f"\nTotal entity types: {total}")
    print(f"Categories: {len(all_entities)}")