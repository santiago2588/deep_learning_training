// Table of Contents functionality
document.addEventListener("DOMContentLoaded", function() {
    // Smooth scrolling for anchor links
    const links = document.querySelectorAll("a[href^=\"#\"]");
    
    links.forEach(link => {
        link.addEventListener("click", function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute("href").substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }
        });
    });

    // Update active TOC item based on scroll position
    const tocLinks = document.querySelectorAll(".toc-list a");
    const sections = document.querySelectorAll("h2[id], h3[id], h4[id]");
    
    function updateActiveTocItem() {
        let current = "";
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            
            if (window.pageYOffset >= sectionTop - 100) {
                current = section.getAttribute("id");
            }
        });
        
        tocLinks.forEach(link => {
            link.classList.remove("active");
            if (link.getAttribute("href") === "#" + current) {
                link.classList.add("active");
            }
        });
    }
    
    // Update on scroll
    window.addEventListener("scroll", updateActiveTocItem);
    
    // Initial update
    updateActiveTocItem();
});

// Add hover effects and animations
document.addEventListener("DOMContentLoaded", function() {
    // Add ripple effect to buttons
    const buttons = document.querySelectorAll(".colab-button, .presentation-button");
    
    buttons.forEach(button => {
        button.addEventListener("click", function(e) {
            const ripple = document.createElement("span");
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + "px";
            ripple.style.left = x + "px";
            ripple.style.top = y + "px";
            ripple.classList.add("ripple");
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Add parallax effect to cards
    const cards = document.querySelectorAll(".card");
    
    window.addEventListener("scroll", function() {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;
        
        cards.forEach((card, index) => {
            const yPos = -(scrolled * (0.1 + index * 0.02));
            card.style.transform = `translateY(${yPos}px)`;
        });
    });
});

// Mobile menu toggle (if needed)
document.addEventListener("DOMContentLoaded", function() {
    const nav = document.querySelector("nav ul");
    const header = document.querySelector("header");
    
    // Add mobile menu button if screen is small
    if (window.innerWidth <= 768) {
        const menuButton = document.createElement("button");
        menuButton.innerHTML = "<i class=\"fas fa-bars\"></i>";
        menuButton.classList.add("mobile-menu-toggle");
        menuButton.style.cssText = `
            background: none;
            border: none;
            color: var(--text-primary);
            font-size: 1.5rem;
            cursor: pointer;
            display: block;
        `;
        
        header.querySelector(".container").appendChild(menuButton);
        
        menuButton.addEventListener("click", function() {
            nav.classList.toggle("mobile-open");
        });
    }
});

// Add CSS for ripple effect
const style = document.createElement("style");
style.textContent = `
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .colab-button, .presentation-button {
        position: relative;
        overflow: hidden;
    }
    
    @media (max-width: 768px) {
        nav ul.mobile-open {
            display: flex !important;
            flex-direction: column;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--header-color);
            padding: 1rem;
            box-shadow: var(--shadow-2);
        }
        
        nav ul {
            display: none;
        }
    }
`;
document.head.appendChild(style);


