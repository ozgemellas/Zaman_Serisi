import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_academic_block_diagram():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Stil Ayarları (Akademik)
    box_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#2c3e50", lw=2)
    arrow_props = dict(arrowstyle="->", color='#2c3e50', lw=2, mutation_scale=20)
    text_props = dict(ha='center', va='center', fontsize=11, color='#2c3e50', fontweight='bold')

    # Fonksiyon: Kutu Çiz
    def draw_box(x, y, w, h, text):
        rect = patches.FancyBboxPatch((x, y), w, h, **box_props)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, **text_props)

    # --- KUTULAR ---
    # 1. Input
    draw_box(0.5, 2.0, 2.0, 1.0, "Ham Veri\n(GPS Sensörleri)")
    
    # 2. Preprocessing
    draw_box(3.5, 2.0, 2.0, 1.0, "Mekânsal Filtre\n&\nFeature Eng.")
    
    # 3. Model Core (Vurgulu)
    # Bunu biraz renkli yapalım
    rect_core = patches.FancyBboxPatch((6.5, 1.5), 2.0, 2.0, boxstyle="round,pad=0.4", fc="#e8f4f8", ec="#2980b9", lw=2)
    ax.add_patch(rect_core)
    ax.text(7.5, 2.5, "Bi-LSTM\nDerin Öğrenme\nModeli", ha='center', va='center', fontsize=11, color='#2980b9', fontweight='bold')
    
    # 4. Output
    draw_box(9.5, 2.0, 2.0, 1.0, "Tahmin Çıktısı\n(t+1, t+3 Saat)")
    
    # --- OKLAR ---
    ax.annotate("", xy=(3.4, 2.5), xytext=(2.6, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6.4, 2.5), xytext=(5.6, 2.5), arrowprops=arrow_props)
    ax.annotate("", xy=(9.4, 2.5), xytext=(8.6, 2.5), arrowprops=arrow_props)
    
    plt.title("Sistem Mimarisi ve Veri Akışı", fontsize=14, pad=20, color='#2c3e50')
    plt.tight_layout()
    plt.savefig("academic_system_diagram.png", dpi=300)
    print("Diyagram oluşturuldu: academic_system_diagram.png")

draw_academic_block_diagram()