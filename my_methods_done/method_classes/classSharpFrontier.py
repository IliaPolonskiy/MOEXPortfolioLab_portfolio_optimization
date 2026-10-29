import numpy as np
import matplotlib.pyplot as plt

class SharpFrontier:
    """
    Класс для аналитического построения эффективной границы 
    и поиска касательного портфеля (Max Sharpe) без ограничений на шорты.
    """
    def __init__(self, mu: np.ndarray, S: np.ndarray) -> None:
        """
        Параметры:
        ----------
        mu : np.ndarray
            Вектор ожидаемых доходностей.
        S : np.ndarray
            Ковариационная матрица.
        """
        self.mu = np.asarray(mu, dtype=np.float64)
        self.S = np.asarray(S, dtype=np.float64)
        self.n = len(self.mu)
        self.ones = np.ones(self.n)

    def get_analytical_max_sharpe(self, rf: float = 0.0) -> dict:
        """
        Аналитический расчёт касательного портфеля (Max Sharpe) для любой rf.
        Шорты разрешены. Единственное ограничение: сумма весов = 1.
        """
        excess = self.mu - rf * self.ones
        
        # Используем solve вместо inv для численной стабильности
        try:
            w_raw = np.linalg.solve(self.S, excess)
        except np.linalg.LinAlgError:
            w_raw = np.linalg.pinv(self.S) @ excess
            
        sum_w = np.sum(w_raw)
        if abs(sum_w) < 1e-12:
            raise ValueError("Сумма весов близка к нулю: невозможно нормировать.")
            
        w = w_raw / sum_w
        
        # Если получился отрицательный Sharpe, переворачиваем портфель
        opt_ret = w @ self.mu
        opt_vol = np.sqrt(w @ self.S @ w)
        opt_sharpe = (opt_ret - rf) / opt_vol if opt_vol > 0 else 0
        
        if opt_sharpe < 0:
            w = -w
            opt_ret = w @ self.mu
            opt_vol = np.sqrt(w @ self.S @ w)
            opt_sharpe = (opt_ret - rf) / opt_vol
            
        return {
            "returns": opt_ret,
            "volatilities": opt_vol,
            "weights": w,
            "sharpe": opt_sharpe
        }

    def get_alpha_beta(self, rf1: float, rf2: float) -> tuple:
        """
        Находит векторы коэффициентов alpha и beta для линейной зависимости весов:
        x = alpha + beta * lambda, где lambda = R / sigma.
        """
        if rf1 == rf2:
            raise ValueError("Безрисковые ставки должны различаться!")

        opt1 = self.get_analytical_max_sharpe(rf=rf1)
        x1 = opt1['weights']
        lambda1 = opt1['returns'] / opt1['volatilities']
        
        opt2 = self.get_analytical_max_sharpe(rf=rf2)
        x2 = opt2['weights']
        lambda2 = opt2['returns'] / opt2['volatilities']
        
        beta = (x1 - x2) / (lambda1 - lambda2)
        alpha = x1 - beta * lambda1
        
        return alpha, beta

    def plot_efficient_frontier(self, ax=None, rf: float = 0.0, max_return_mult: float = 1.5):
        """
        Строит гладкую аналитическую эффективную границу (без ограничений).
        А также точку касания и CML для заданной rf.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # --- 1. Построение аналитической границы (гиперболы) ---
        # Формула для неограниченной границы: w = g + h * R
        try:
            inv_S = np.linalg.inv(self.S)
        except np.linalg.LinAlgError:
            inv_S = np.linalg.pinv(self.S)
            
        A = self.ones.T @ inv_S @ self.ones
        B = self.ones.T @ inv_S @ self.mu
        C = self.mu.T @ inv_S @ self.mu
        D = A * C - B**2
        
        g = (C * (inv_S @ self.ones) - B * (inv_S @ self.mu)) / D
        h = (A * (inv_S @ self.mu) - B * (inv_S @ self.ones)) / D
        
        # Генерируем диапазон целевых доходностей от минимального риска до максимума
        min_var_ret = B / A # Доходность портфеля с глобальным минимальным риском
        max_ret = np.max(self.mu) * max_return_mult
        target_returns = np.linspace(min_var_ret, max_ret, 100)
        
        # Считаем веса, риски и доходности для каждой точки
        risks = []
        rets = []
        for R in target_returns:
            w = g + h * R
            rets.append(w @ self.mu)
            risks.append(np.sqrt(w @ self.S @ w))
            
        ax.plot(risks, rets, 'b-', label='Эффективная граница (Без ограничений)', linewidth=2)
        
        # --- 2. Построение оптимального портфеля и CML ---
        opt = self.get_analytical_max_sharpe(rf=rf)
        
        # Точка касания
        ax.scatter([opt['volatilities']], [opt['returns']], 
                   marker='*', s=300, c='gold', edgecolors='black', zorder=6, 
                   label=f'Max Sharpe (rf={rf*100}%)')
        
        # Линия CML
        x_max = max(risks) * 1.1
        x_cml = np.linspace(0, x_max, 100)
        y_cml = rf + opt['sharpe'] * x_cml
        ax.plot(x_cml, y_cml, 'g--', linewidth=1.5, label='CML')
        
        # Безрисковая ставка
        ax.scatter([0], [rf], marker='o', s=80, c='green', zorder=6, label='Безрисковый актив')
        
        ax.set_xlabel('Волатильность (σ)')
        ax.set_ylabel('Ожидаемая доходность')
        ax.set_title('Аналитическая эффективная граница и CML')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax