import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import CLA

class CLAFrontier:
    def __init__(self, mu, S):
        """
        Инициализация и расчет базовой эффективной границы.
        mu: массив ожидаемых доходностей
        S: ковариационная матрица
        """
        self.mu = np.asarray(mu, dtype=np.float64)
        self.S = np.asarray(S, dtype=np.float64)
        
        # Обучаем классический CLA
        self.cla = CLA(self.mu, self.S)
        
        # Получаем сырые точки границы
        self.raw_returns, self.raw_volatilities, self.raw_weights = self.cla.efficient_frontier()

    def get_filtered_turning_points(self, tol=1e-4):
        """
        Задача 1: Возвращает отфильтрованные угловые портфели.
        Оставляет только те, где меняется активный набор.
        Возвращает DataFrame с колонками: returns, volatilities, weights.
        """
        weights_arr = np.asarray(self.raw_weights, dtype=np.float64)
        returns_arr = np.asarray(self.raw_returns, dtype=np.float64)
        volatilities_arr = np.asarray(self.raw_volatilities, dtype=np.float64)
        
        if len(weights_arr) == 0:
            return pd.DataFrame(columns=['returns', 'volatilities', 'weights'])

        f_returns, f_volatilities, f_weights = [], [], []
        
        def get_active_mask(w):
            return (w > tol) & (w < 1 - tol)

        current_mask = get_active_mask(weights_arr[0])
        f_returns.append(returns_arr[0])
        f_volatilities.append(volatilities_arr[0])
        f_weights.append(weights_arr[0])

        for i in range(1, len(weights_arr)):
            new_mask = get_active_mask(weights_arr[i])
            if not np.array_equal(current_mask, new_mask):
                f_returns.append(returns_arr[i])
                f_volatilities.append(volatilities_arr[i])
                f_weights.append(weights_arr[i])
                current_mask = new_mask

        # Гарантируем, что последняя точка попала в итог
        if not np.array_equal(f_weights[-1], weights_arr[-1]):
            f_returns.append(returns_arr[-1])
            f_volatilities.append(volatilities_arr[-1])
            f_weights.append(weights_arr[-1])

        return pd.DataFrame({
            'returns': f_returns,
            'volatilities': f_volatilities,
            'weights': f_weights
        })

    def get_optimal_portfolio(self):
        """
        Задача 3: Возвращает параметры оптимального портфеля (с rf=0).
        Возвращает словарь с доходностью, волатильностью, весами и Sharpe.
        """
        # max_sharpe() возвращает словарь весов
        weights_dict = self.cla.max_sharpe()
        weights = np.array(list(weights_dict.values()))
        
        # Считаем метрики через встроенный метод (risk_free_rate по умолчанию 0.0 в CLA)
        opt_ret, opt_vol, max_sharpe_ratio = self.cla.portfolio_performance(verbose=False)
        
        return {
            "returns": opt_ret,
            "volatilities": opt_vol,
            "weights": weights,
            "sharpe": max_sharpe_ratio
        }

    def plot_frontier(self, ax=None):
        """
        Задача 2: Строит график доходности от волатильности с отфильтрованными портфелями.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        df_filtered = self.get_filtered_turning_points()
        
        # Рисуем гладкую линию по всем точкам (как в вашем ноутбуке)
        ax.plot(self.raw_volatilities, self.raw_returns, 'b-', alpha=0.5, label='Все точки CLA')
        
        # Рисуем отфильтрованные точки
        ax.scatter(df_filtered['volatilities'], df_filtered['returns'], 
                   marker='D', s=30, c='pink', edgecolors='black', zorder=5, 
                   label=f'Угловые портфели ({len(df_filtered)} шт.)')
        
        ax.set_xlabel('Волатильность (σ)')
        ax.set_ylabel('Ожидаемая доходность')
        ax.set_title('Эффективная граница (CLA)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax

    def plot_cml(self, ax=None):
        """
        Задача 4: Строит CML с нулевой безрисковой ставкой и оптимальный портфель.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        opt = self.get_optimal_portfolio()
        rf = 0.0
        
        # Рисуем точку касания (оптимальный портфель)
        ax.scatter([opt['volatilities']], [opt['returns']], 
                   marker='*', s=200, c='gold', edgecolors='black', zorder=6, 
                   label=f'Optimal (Sharpe = {opt["sharpe"]:.2f})')
        
        # Рисуем линию CML
        x_max = max(self.raw_volatilities) * 1.1
        x_cml = np.linspace(0, x_max, 100)
        y_cml = rf + opt['sharpe'] * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, label='CML (rf=0)')
        
        # Безрисковая ставка
        ax.scatter([0], [rf], marker='o', s=80, c='green', zorder=6, label='Безрисковый актив')
        
        ax.set_xlabel('Волатильность (σ)')
        ax.set_ylabel('Ожидаемая доходность')
        ax.set_title('Линия рынка капитала (CML)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax