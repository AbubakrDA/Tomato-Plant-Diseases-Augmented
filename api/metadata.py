"""Domain metadata, disease descriptions, and treatment recommendations."""

DISEASE_METADATA = {
    "Tomato___Bacterial_spot": {
        "display_name": "Bacterial Spot",
        "scientific_agent": "Xanthomonas perforans / euvesicatoria",
        "is_healthy": False,
        "symptoms": "Small, water-soaked, circular to irregular dark brown spots with a greasy appearance on leaves and stems.",
        "treatment_recommendation": (
            "Apply copper-based bactericides early in the disease cycle. Avoid overhead irrigation "
            "to prevent water splashing. Practice crop rotation with non-solanaceous crops for at least two years."
        )
    },
    "Tomato___Early_blight": {
        "display_name": "Early Blight",
        "scientific_agent": "Alternaria solani",
        "is_healthy": False,
        "symptoms": "Concentric rings ('target-like' bullseye pattern) on older lower leaves, often surrounded by a yellow halo.",
        "treatment_recommendation": (
            "Prune lower infected foliage and apply appropriate fungicides (such as chlorothalonil or copper). "
            "Maintain soil mulching to prevent soil spores from splashing onto foliage."
        )
    },
    "Tomato___Late_blight": {
        "display_name": "Late Blight",
        "scientific_agent": "Phytophthora infestans",
        "is_healthy": False,
        "symptoms": "Large, irregular water-soaked lesions that turn dark brown/purplish, often with white fungal mold underneath in humid conditions.",
        "treatment_recommendation": (
            "Urgent action required: remove and destroy severely infected plants immediately. "
            "Apply targeted systemic fungicides preventative to humid, cool weather. Ensure wide spacing for air circulation."
        )
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Leaf Mold",
        "scientific_agent": "Passalora fulva (Cladosporium fulvum)",
        "is_healthy": False,
        "symptoms": "Pale greenish-yellow spots on upper leaf surfaces and velvety olive-green to brown mold on the undersides.",
        "treatment_recommendation": (
            "Reduce humidity below 85% by improving greenhouse ventilation. Increase plant spacing "
            "and prune lower sucker leaves. Apply preventative fungicides if high humidity persists."
        )
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Septoria Leaf Spot",
        "scientific_agent": "Septoria lycopersici",
        "is_healthy": False,
        "symptoms": "Numerous small, circular lesions (1-3 mm) with gray or tan centers and distinct dark brown borders.",
        "treatment_recommendation": (
            "Remove infected lower leaves promptly. Mulch around plants to block spore splashback. "
            "Apply bio-fungicides or chlorothalonil at the first onset of spotting."
        )
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Two-Spotted Spider Mites",
        "scientific_agent": "Tetranychus urticae",
        "is_healthy": False,
        "symptoms": "Yellowish stippling/flecking on leaves accompanied by fine webbing on undersides, leading to bronzing and leaf drop.",
        "treatment_recommendation": (
            "Spray with horticultural oil, neem oil, or insecticidal soap, paying close attention to leaf undersides. "
            "Release predatory beneficial mites (Phytoseiulus persimilis) in controlled environments."
        )
    },
    "Tomato___Target_Spot": {
        "display_name": "Target Spot",
        "scientific_agent": "Corynespora cassiicola",
        "is_healthy": False,
        "symptoms": "Brown circular lesions with faint concentric rings, often causing significant defoliation in warm, humid weather.",
        "treatment_recommendation": (
            "Improve canopy air circulation. Avoid excessive nitrogen fertilization. "
            "Apply protective fungicides (e.g., strobilurin or triazole group) early during canopy development."
        )
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "scientific_agent": "Begomovirus (transmitted by Whiteflies)",
        "is_healthy": False,
        "symptoms": "Severe upward leaf curling, yellowing of leaf margins, reduced leaf size, and stunted plant growth.",
        "treatment_recommendation": (
            "Manage whitefly vectors using yellow sticky cards, reflective mulches, and appropriate insecticides. "
            "Rogue and safely destroy infected plants immediately to prevent transmission to nearby crops."
        )
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato Mosaic Virus (ToMV)",
        "scientific_agent": "Tobamovirus",
        "is_healthy": False,
        "symptoms": "Mottled light and dark green patterns on leaves, leaf distortion (fern-like appearance), and plant stunting.",
        "treatment_recommendation": (
            "ToMV is easily transmitted mechanically: disinfect all pruning tools in 10-20% bleach or milk solution. "
            "Wash hands thoroughly and avoid handling healthy plants after touching suspect ones."
        )
    },
    "Tomato___healthy": {
        "display_name": "Healthy Tomato Leaf",
        "scientific_agent": "None (Healthy)",
        "is_healthy": True,
        "symptoms": "Vibrant green leaves with uniform coloration, smooth texture, and no visible lesions or discoloration.",
        "treatment_recommendation": (
            "Maintain balanced watering, optimal sunlight exposure, consistent soil nutrition, "
            "and routine inspections to keep foliage thriving."
        )
    }
}

