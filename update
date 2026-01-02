import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="Multivariable Calculus Explorer", layout="wide")
st.markdown("""
<style>
.stApp { background-color: #111111; color: #FFFFFF; font-family: 'Segoe UI', sans-serif;}
h1 {color:#00FFFF;}
h2 {color:#00FFAA;}
.stTextInput>div>div>input { color: black; background-color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Multivariable Calculus Explorer")

# ---------------------------
# Variables & intervals
# ---------------------------
var_choice = st.selectbox("Number of variables:", ["2 variables (f(x,y))", "3 variables (f(x,y,z))"])
if var_choice.startswith("2"):
    var_symbols = sp.symbols("x y")
else:
    var_symbols = sp.symbols("x y z")

col1, col2 = st.columns(2)
with col1:
    x_min, x_max = st.number_input("x min", value=-5.0), st.number_input("x max", value=5.0)
with col2:
    y_min, y_max = st.number_input("y min", value=-5.0), st.number_input("y max", value=5.0)

if var_choice.startswith("3"):
    z_min, z_max = st.number_input("z min", value=-5.0), st.number_input("z max", value=5.0)
    z0 = st.slider("Fix z value for 3D slice", min_value=float(z_min), max_value=float(z_max), value=0.0)

# ---------------------------
# Function Input
# ---------------------------
st.subheader("📝 Enter Your Function (use ^ for powers)")

# Session state for calculator keyboard
if "func_str" not in st.session_state:
    st.session_state.func_str = ""

# Display current function
st.text_input("Function:", key='func_str', value=st.session_state.func_str)

# Calculator keyboard (like Desmos)
buttons = ["x", "y", "z", "+", "-", "*", "/", "^", "(", ")", 
           "sin(", "cos(", "tan(", "exp(", "log(", "sqrt(", "CLEAR"]
cols = st.columns(len(buttons))
for i, btn in enumerate(buttons):
    if cols[i].button(btn):
        if btn == "CLEAR":
            st.session_state.func_str = ""
        else:
            st.session_state.func_str += btn

func_input = st.session_state.func_str.replace("^", "**")

# Allowed functions
allowed_funcs = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                 "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}

# ---------------------------
# Parse function
# ---------------------------
try:
    f = sp.sympify(func_input, locals=allowed_funcs)
except Exception as e:
    st.error(f"❌ Invalid function. Error: {e}")
    st.stop()

# ---------------------------
# Grid for plotting
# ---------------------------
grid_vals_x = np.linspace(x_min, x_max, 50)
grid_vals_y = np.linspace(y_min, y_max, 50)
X, Y = np.meshgrid(grid_vals_x, grid_vals_y)

if var_choice.startswith("3"):
    f_eval = f.subs('z', z0)
else:
    f_eval = f

try:
    f_lamb = sp.lambdify(var_symbols[:2], f_eval, "numpy")
    Z = f_lamb(X, Y)
except Exception as e:
    st.error(f"⚠️ Cannot evaluate function on this domain: {e}")
    st.stop()

# ---------------------------
# Plotting function
# ---------------------------
def plot_surface(X, Y, Z, title="z = f(x,y)"):
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])
    fig.update_layout(
        title=title,
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        font=dict(color='white'),
        scene=dict(
            xaxis=dict(title='x', backgroundcolor="#111111", gridcolor="gray", color="white"),
            yaxis=dict(title='y', backgroundcolor="#111111", gridcolor="gray", color="white"),
            zaxis=dict(title='z', backgroundcolor="#111111", gridcolor="gray", color="white")
        ),
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# ---------------------------
# Calculus quantities
# ---------------------------
partials = [sp.diff(f, s) for s in var_symbols[:2]]
diff_text = " + ".join([f"({p})d{s}" for p, s in zip(partials, var_symbols[:2])])
sample_point = {var_symbols[0]: 1, var_symbols[1]: 1}
if var_choice.startswith("3"):
    sample_point[var_symbols[2]] = z0

# ---------------------------
# Show all features together
# ---------------------------
st.subheader("🔹 Surface Plot")
st.plotly_chart(plot_surface(X, Y, Z, title="Surface Plot: z=f(x,y)"), use_container_width=True)

# Meaning
st.subheader("🔹 Meaning of the Function")
z_val = f.subs(sample_point)
st.markdown(f"""
- Function: `{func_input}`  
- At point (x=1, y=1): z = `{z_val}`  
- Each (x,y) gives height z forming a 3D surface
""")

# Partial Derivatives
st.subheader("🔹 Partial Derivatives")
for s, p in zip(var_symbols[:2], partials):
    st.markdown(f"∂f/∂{s} = `{p}` → At (1,1): {p.subs(sample_point)}")

# Gradient
st.subheader("🔹 Gradient")
grad_val = [p.subs(sample_point) for p in partials]
st.markdown(f"Gradient vector at (1,1): {grad_val}")

# Differentials
st.subheader("🔹 Differentials")
delta_x, delta_y = 0.1, 0.1
df_val = sum([p.subs(sample_point)*dx for p, dx in zip(partials, [delta_x, delta_y])])
st.markdown(f"df ≈ {diff_text} → At (1,1) with dx={delta_x}, dy={delta_y}: df ≈ {df_val}")

# Critical Points (only for 2-variable)
if var_choice.startswith("2"):
    st.subheader("🔹 Critical Points")
    x, y = var_symbols
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    try:
        crit_points = sp.solve([fx, fy], (x, y), dict=True)
        if crit_points:
            st.write(f"Found {len(crit_points)} critical point(s):")
            for pt in crit_points:
                fxx = sp.diff(fx, x)
                fyy = sp.diff(fy, y)
                fxy = sp.diff(fx, y)
                H = fxx*fyy - fxy**2
                H_val = H.subs(pt)
                f_val = f.subs(pt)
                if H_val > 0:
                    kind = "Local Min" if fxx.subs(pt) > 0 else "Local Max"
                elif H_val < 0:
                    kind = "Saddle Point"
                else:
                    kind = "Cannot Determine"
                st.write(f"- Point {pt}: {kind}, f={f_val}")
        else:
            st.write("No critical points found.")
    except Exception as e:
        st.write(f"Could not compute critical points: {e}")
