import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Multivariable Functions Explorer")
st.title("📊 Interactive Multivariable Functions Explorer")

st.write(
    "This app helps you explore functions of several variables using interactive graphs. "
    "You can rotate, zoom, and analyze how the surface behaves."
)

# ---------------------------------------------------
# Variable selection
# ---------------------------------------------------
var_type = st.selectbox(
    "Select function type:",
    ["Two variables: f(x, y)", "Three variables: f(x, y, z)"]
)

# Define symbols
x, y, z = sp.symbols("x y z")

# ---------------------------------------------------
# Function input
# ---------------------------------------------------
default_func = "sin(x) + cos(y)" if var_type.startswith("Two") else "x**2 + y**2 + z**2"
func_input = st.text_input("Enter your function:", default_func)

# Allowed math functions
allowed_funcs = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt
}

# ---------------------------------------------------
# Safe parsing with friendly error messages
# ---------------------------------------------------
try:
    f = sp.sympify(func_input, locals=allowed_funcs)
except Exception as e:
    st.error("❌ Invalid function input.")
    st.markdown(
        """
        **Please check the following:**
        - Use `**` for powers (example: `x**2`)
        - Use `sin(x)`, not `sin x`
        - Use `log(x)` for natural logarithm
        - Allowed variables: x, y, z
        
        **Examples of valid inputs:**
        - `sin(x) + cos(y)`
        - `x**2 + y**2`
        - `exp(x*y)`
        - `x**2 + y**2 + z**2`
        """
    )
    st.stop()

# ---------------------------------------------------
# Topic selection
# ---------------------------------------------------
topic = st.selectbox(
    "Choose a topic:",
    [
        "Meaning & Visualization",
        "Partial Derivatives",
        "Gradient & Steepest Ascent",
        "Differentials"
    ]
)

# ---------------------------------------------------
# Domain grid
# ---------------------------------------------------
vals = np.linspace(-5, 5, 60)
X, Y = np.meshgrid(vals, vals)

# ---------------------------------------------------
# Handle 3-variable slicing
# ---------------------------------------------------
if var_type.startswith("Three"):
    z0 = st.slider("Fix z value (slice):", -5.0, 5.0, 0.0)
    f_eval = f.subs(z, z0)
else:
    f_eval = f

# Convert to numeric function
try:
    f_lamb = sp.lambdify((x, y), f_eval, "numpy")
    Z = f_lamb(X, Y)
except:
    st.error(
        "⚠️ The function cannot be evaluated on the selected domain.\n\n"
        "This may happen due to division by zero or invalid values (e.g. log(x) for x ≤ 0)."
    )
    st.stop()

# ---------------------------------------------------
# Plot function
# ---------------------------------------------------
def plot_surface(X, Y, Z):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# ---------------------------------------------------
# Explanation based on surface
# ---------------------------------------------------
def surface_explanation(Z):
    if np.nanmin(Z) >= 0:
        return "The surface lies above the xy-plane, suggesting a minimum point."
    elif np.nanmax(Z) <= 0:
        return "The surface lies below the xy-plane, suggesting a maximum point."
    else:
        return (
            "The surface has both positive and negative values. "
            "This indicates varying curvature or a possible saddle point."
        )

# ---------------------------------------------------
# 1. Visualization
# ---------------------------------------------------
if topic == "Meaning & Visualization":
    st.subheader("🔹 Meaning & Visualization")

    st.write(
        "The graph represents the surface **z = f(x, y)**. "
        "Rotate and zoom the graph to understand how the function behaves."
    )

    st.plotly_chart(plot_surface(X, Y, Z), use_container_width=True)
    st.info(surface_explanation(Z))

# ---------------------------------------------------
# 2. Partial Derivatives
# ---------------------------------------------------
elif topic == "Partial Derivatives":
    st.subheader("🔹 Partial Derivatives")

    fx = sp.diff(f_eval, x)
    fy = sp.diff(f_eval, y)

    st.latex(r"\frac{\partial f}{\partial x} = " + sp.latex(fx))
    st.latex(r"\frac{\partial f}{\partial y} = " + sp.latex(fy))

    x0 = st.number_input("x₀:", value=1.0)
    y0 = st.number_input("y₀:", value=1.0)

    st.write(
        "Partial derivatives describe how the surface changes "
        "when one variable changes while the other is held constant."
    )

    st.plotly_chart(plot_surface(X, Y, Z), use_container_width=True)

# ---------------------------------------------------
# 3. Gradient
# ---------------------------------------------------
elif topic == "Gradient & Steepest Ascent":
    st.subheader("🔹 Gradient & Steepest Ascent")

    fx = sp.diff(f_eval, x)
    fy = sp.diff(f_eval, y)

    st.latex(r"\nabla f = \left(" + sp.latex(fx) + ", " + sp.latex(fy) + r"\right)")

    st.write(
        "The gradient vector points in the direction of steepest ascent "
        "on the surface."
    )

    st.plotly_chart(plot_surface(X, Y, Z), use_container_width=True)

# ---------------------------------------------------
# 4. Differentials
# ---------------------------------------------------
elif topic == "Differentials":
    st.subheader("🔹 Differentials")

    fx = sp.diff(f_eval, x)
    fy = sp.diff(f_eval, y)

    dx = st.number_input("dx:", value=0.1)
    dy = st.number_input("dy:", value=0.1)

    st.latex(r"df = f_x dx + f_y dy")

    st.write(
        "The differential provides a linear approximation of how the "
        "function value changes near a point."
    )

    st.plotly_chart(plot_surface(X, Y, Z), use_container_width=True)
