import unittest
import pandas as pd
from Stat_bivar.Stat_bivar import StatBivarPlot

class TestStatBivarPlot(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        data = {
            'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'y': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        }
        df = pd.DataFrame(data)
        self.stat_bivar_plot = StatBivarPlot(df, colonne_x='x', colonne_y='y', x_type='QuantiContinu', y_type='QuantiContinu')

    def test_annalyse_biv_default(self):
        # Test with default parameters
        self.stat_bivar_plot.annalyse_biv()

    def test_annalyse_biv_boxplot(self):
        # Test with figure_droite set to 'boxplot'
        self.stat_bivar_plot.annalyse_biv(figure_droite='boxplot')

    def test_annalyse_biv_heatmap(self):
        # Test with figure_droite set to 'heatmap'
        self.stat_bivar_plot.annalyse_biv(figure_droite='heatmap')

    def test_annalyse_biv_scatterplot(self):
        # Test with figure_droite set to 'scatterplot'
        self.stat_bivar_plot.annalyse_biv(figure_droite='scatterplot')

if __name__ == '__main__':
    unittest.main()
