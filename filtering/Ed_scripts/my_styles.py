import matplotlib.pyplot as plt

def set_ppt_style():
    plt.style.use('bmh')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    plt.rcParams['grid.color'] = 'black'
    plt.rcParams['legend.shadow'] = True
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.figsize'] = 6.4, 4.8   # figure size in inches
    plt.rcParams['figure.dpi'] = 150

#    plt.rcParams['grid.color']    = 'black'    # grid color
#    plt.rcParams['grid.linestyle'] = '--'         # solid
#    plt.rcParams['grid.linewidth'] = 0.8       # in points
#    plt.rcParams['grid.alpha'] =  0.5       # transparency, between 0.0 and 1.0
#    
def set_paper_style():
    plt.style.use('classic')
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'serif'

    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['figure.titlesize'] = 14

    plt.rcParams['legend.numpoints'] = 1

    plt.rcParams['figure.figsize'] = 6.4, 5.0   # figure size in inches
