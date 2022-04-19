import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import cvgutils.Viz as cvgviz
import argparse

parser = argparse.ArgumentParser()
parser = cvgviz.logger.parse_arguments(parser)
opts = parser.parse_args()

logger = cvgviz.logger(opts)

im = np.random.rand(1,100,100,3)

logger.addImage([im, im],[r'$hi~there~\alpha \beta \gamma$','im2'],'img',dim_type='BHWC')
# fig = make_subplots(rows=1, cols=2,subplot_titles=("Plot 1", "Plot 2"))

# fig.add_trace(px.imshow(im).data[0], row=1, col=1)
# fig.add_trace(px.imshow(im).data[0], row=1, col=2)

# fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")

# fig.write_image("fig1.png")