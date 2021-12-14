import matplotlib.pyplot as plt
def Color_palette(k):
    """Takes integer, Returns color scheme."""
    Color_schemes1 = [
        [i['color'] for i in plt.rcParams['axes.prop_cycle']],
        [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
         u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'],
        ["#70d6ff", "#ff70a6", "#ff9770", "#ffd670", "#e9ff70"],
        ['#F61613', '#EECC86', '#34AE8E', '#636253', '#A26251'],
        ["#8acdea", "#746d75", "#8c4843", "#9e643c", "#ede6f2"],
        ['#2CBDFE', '#47DBCD', '#9D2EC5',  '#F3A0F2', '#661D98', '#F5B14C'],
        ['#845EC2', '#ffc75f', '#f9f871', '#ff5e78'],
        ['#fff3e6', '#1a508b', '#0d335d', '#c1a1d3'],
        ["#f72585", "#b5179e", "#7209b7", "#560bad", "#480ca8",
         "#3a0ca3", "#3f37c9", "#4361ee", "#4895ef", "#4cc9f0"],
        ["#22223b", "#4a4e69", "#9a8c98", "#c9ada7", "#f2e9e4"],
        ["#d8f3dc", "#b7e4c7", "#95d5b2", "#74c69d",
         "#52b788", "#40916c", "#2d6a4f", "#1b4332", "#081c15"],
        ["#ffbe0b", "#fb5607", "#ff006e", "#8338ec", "#3a86ff"],
        ["#7400b8", "#6930c3", "#5e60ce", "#5390d9", "#4ea8de",
         "#48bfe3", "#56cfe1", "#64dfdf", "#72efdd", "#80ffdb"],
        ["#cc8b86", "#f9eae1", "#7d4f50", "#d1be9c", "#aa998f"],
        ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"],
        ["#8c1c13", "#bf4342", "#e7d7c1", "#a78a7f", "#735751"],
        ["#f72585", "#b5179e", "#7209b7", "#560bad", "#480ca8",
         "#3a0ca3", "#3f37c9", "#4361ee", "#4895ef", "#4cc9f0"],
        ["#006466", "#065a60", "#0b525b", "#144552", "#1b3a4b",
         "#212f45", "#272640", "#312244", "#3e1f47", "#4d194d"]]
    Color_schemes = Color_schemes1
    for i in range(len(Color_schemes1)):
        Color_schemes.append(reversed(Color_schemes1[i]))
    return Color_schemes[k]
