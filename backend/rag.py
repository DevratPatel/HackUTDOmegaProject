# Imports
import requests

# Prompting Llama for a response using SambaNova API
def get_recommendation(prev_msgs):

    samba_nova_url = "https://api.sambanova.ai/v1/chat/completions"
    samba_nova_key = "4c08c7f6-63fe-4e36-bb86-dc042578a025"

    prompt = f"""
        You are a sales representative for Frontier. Your goal is to recommend the top 3 best internet products based solely on the user's provided information. Recommend products in the following structured format:

        **Recommendation Format:**
        ```json
        {{
            "products": ["<name_of_recommended_product_1>", "<name_of_recommended_product_2>", "<name_of_recommended_product_3>"]
        }}
        ```

        Products available:
        1. **Fiber 500** - 500Mbps, standard router ($45/mo)  
        2. **Fiber 1 Gig** - 1Gbps, standard router ($65/mo)  
        3. **Fiber 2 Gig** - 2Gbps, upgraded router + 1 extender ($99/mo)  
        4. **Fiber 5 Gig** - 5Gbps, premium router ($129/mo)  
        5. **Fiber 7 Gig** - 7Gbps, premium router + 1 extender ($299/mo)  
        6. **Additional Extender** - Extra extenders for any plan ($5/mo per extender)  
        7. **Whole Home Wi-Fi** - Up to 2 extenders for Fiber 2 Gig or below, 1 extender for Fiber 5/7 Gig ($10/mo)  
        8. **Unbreakable Wi-Fi** - Backup internet during outages ($25/mo)  
        9. **Battery Backup** - 4 hours of power for Unbreakable Wi-Fi ($130 one-time)  
        10. **Wi-Fi Security** - Advanced network protection ($5/mo)  
        11. **Wi-Fi Security Plus** - Includes Wi-Fi Security, VPN, and Password Manager ($10/mo)  
        12. **Total Shield** - Security for up to 10 devices ($10/mo)  
        13. **My Premium Tech Pro** - Premium tech support ($10/mo)  
        14. **Identity Protection** - Personal data monitoring and theft insurance ($10/mo)  
        15. **YouTube TV** - 100+ live channels, no set-top box required ($79.99/mo)  

        Based on the user's needs and details provided in a single input, return up to 3 top products. Do not ask follow-up questions. Provide your response in the structured format immediately.
        
        Previous Messages: {prev_msgs}
        """

    response = requests.post(
        samba_nova_url,
        headers={
            "Authorization": f"Bearer {samba_nova_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "Meta-Llama-3.1-70B-Instruct",
            "messages": [
                {"role": "system", "content": "You are an ISP sales expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
    )

    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"]
        return result
    else:
        return f"Error: {response.status_code} - {response.text}"

