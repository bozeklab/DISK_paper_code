num_keypoints = 20
keypoints = ['HeadF', 'HeadB', 'HeadL',
             'SpineF', 'SpineM', 'SpineL',
             'Offset1', 'Offset2',
             'HipL', 'HipR',
              'ElbowL', 'ArmL', 'ShoulderL',
              'ShoulderR', 'ElbowR', 'ArmR',
              'KneeR', 'KneeL', 'ShinL', 'ShinR']
center = 4
original_directory = '/projects/ag-bozek/france/results_behavior/datasets/Fish_v3_60stride60'
neighbor_links = [(0, 1), (0, 2), (1, 2),  # head links
        (1, 3), (3, 4), (4, 5),  # head to spine + spin links
        (3, 6), (4, 6), (4, 7), (6, 7), (5, 7),  # links to offset
        (5, 8), (5, 9),  # spin to hips
        (3, 12), (12, 10), (11, 10),  # left arm
        (3, 13), (13, 14), (14, 15),  # right arm
        (9, 16), (16, 19),  # right leg
        (8, 17), (17, 18)  # left leg
        ]
link_colors =  ['orange', 'orange', 'orange',  # head links
        'gold', 'gold', 'gold',  # head to spine + spin links
        'grey', 'grey', 'grey', 'grey', 'grey',  # links to offset
        'gold', 'gold',   # spin to hips
        'cornflowerblue', 'cornflowerblue', 'cornflowerblue',  # left arm
        'turquoise', 'turquoise', 'turquoise',  # right arm
         'hotpink', 'hotpink',   # right leg
         'purple', 'purple',  # left leg
        ]
