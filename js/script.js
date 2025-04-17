// Aguardar que o DOM esteja completamente carregado
document.addEventListener('DOMContentLoaded', function() {
    // Menu responsivo
    const menuToggle = document.querySelector('.menu-toggle');
    const menu = document.querySelector('.menu');

    if (menuToggle && menu) {
        menuToggle.addEventListener('click', function() {
            menu.classList.toggle('active');
        });

        // Fechar menu ao clicar em um link
        const menuLinks = document.querySelectorAll('.menu a');
        menuLinks.forEach(link => {
            link.addEventListener('click', function() {
                menu.classList.remove('active');
            });
        });
    }

    // Rolagem suave para links âncora
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                // Adicionar offset para compensar a altura do header fixo
                const headerHeight = document.querySelector('header').offsetHeight;
                const targetPosition = targetElement.getBoundingClientRect().top + window.pageYOffset - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Tratamento do formulário de contato
    const contatoForm = document.getElementById('contato-form');
    if (contatoForm) {
        contatoForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const nome = document.getElementById('nome').value;
            const email = document.getElementById('email').value;
            const mensagem = document.getElementById('mensagem').value;
            
            // Desativar o botão de envio e mostrar indicador de carregamento
            const submitButton = contatoForm.querySelector('button[type="submit"]');
            const originalButtonText = submitButton.textContent;
            submitButton.disabled = true;
            submitButton.textContent = 'Enviando...';
            
            try {
                // Enviar dados para o servidor
                const response = await fetch('/api/contato', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ nome, email, mensagem })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Mostrar mensagem de sucesso
                    alert(data.message);
                    // Limpar o formulário
                    contatoForm.reset();
                } else {
                    // Mostrar mensagem de erro
                    alert('Erro ao enviar o formulário. Por favor, tente novamente.');
                }
            } catch (error) {
                console.error('Erro:', error);
                alert('Erro ao enviar o formulário. Por favor, tente novamente mais tarde.');
            } finally {
                // Reativar o botão de envio
                submitButton.disabled = false;
                submitButton.textContent = originalButtonText;
            }
        });
    }

    // Efeito de revelar elementos ao rolar a página
    function revealOnScroll() {
        const revealElements = document.querySelectorAll('.servico-card, .section-title');
        
        revealElements.forEach(element => {
            const windowHeight = window.innerHeight;
            const elementTop = element.getBoundingClientRect().top;
            const elementVisible = 150;
            
            if (elementTop < windowHeight - elementVisible) {
                element.classList.add('active');
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            } else {
                element.style.opacity = '0';
                element.style.transform = 'translateY(20px)';
            }
        });
    }

    // Adicionar classe para animação de fade-in nos elementos
    document.querySelectorAll('.servico-card, .section-title').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease-in-out, transform 0.6s ease-in-out';
    });

    // Chamar a função ao carregar a página e ao rolar
    window.addEventListener('scroll', revealOnScroll);
    revealOnScroll(); // Chamar uma vez ao carregar
}); 